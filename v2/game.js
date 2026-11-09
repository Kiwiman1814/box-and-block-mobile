// Hand Rehab v2 - game half.
// Consumes the tracker API from CONTRACT.md: createTracker() -> detect(video, nowMs) -> HandResult|null.
// tracker.js is imported lazily (only for "Start with camera") so touch test mode works without it.

const $ = id => document.getElementById(id);
const cv = $("game"), g = cv.getContext("2d"), video = $("video");
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ---------------------------------------------------------------------------
// State (exposed as window.__game.state)
// ---------------------------------------------------------------------------
const state = {
  screen: "start",          // start | menu | play | results
  auto: true,               // false = rAF/watchdog stop calling step(); drive frames via __game.step(t)
  useCamera: false,         // false = touch test mode (MockTracker)
  facing: "user",           // requested camera
  mirror: false,            // draw/map mirrored (front camera)
  stream: null, tracker: null, trackerPromise: null, mock: null,
  res: null,                // last HandResult (or null)
  hand: { seen: false, x: 0, y: 0, closure: 0, gesture: null, lastSeen: -1e9 },
  level: null, levelNo: 0,
  flash: { text: "", until: 0 },
  lastNow: 0, lastDetect: 0, lastVideoTime: -1, detectErr: null,
  W: 0, H: 0, DPR: 1, inset: { t: 0, r: 0, b: 0, l: 0 },
};

// ---------------------------------------------------------------------------
// Layout / resize (also reads safe-area insets for canvas-drawn content)
// ---------------------------------------------------------------------------
const probe = document.createElement("div");
probe.style.cssText = "position:fixed;left:0;top:0;visibility:hidden;pointer-events:none;" +
  "padding:env(safe-area-inset-top,0px) env(safe-area-inset-right,0px) env(safe-area-inset-bottom,0px) env(safe-area-inset-left,0px)";
document.body.appendChild(probe);

function resize() {
  state.DPR = Math.min(2, window.devicePixelRatio || 1);
  state.W = window.innerWidth; state.H = window.innerHeight;
  cv.width = Math.round(state.W * state.DPR); cv.height = Math.round(state.H * state.DPR);
  const cs = getComputedStyle(probe);
  state.inset = { t: parseFloat(cs.paddingTop) || 0, r: parseFloat(cs.paddingRight) || 0,
                  b: parseFloat(cs.paddingBottom) || 0, l: parseFloat(cs.paddingLeft) || 0 };
}
window.addEventListener("resize", resize);
window.addEventListener("orientationchange", () => setTimeout(resize, 200));
if (window.visualViewport) window.visualViewport.addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------------------
// Source <-> screen mapping. One transform for everything: camera picture,
// skeleton, palm cursor, and the touch MockTracker (which converts back).
// Source = camera frame (camera mode) or the screen itself (touch mode).
// ---------------------------------------------------------------------------
function srcSize() {
  if (state.useCamera && video.videoWidth && video.videoHeight) return { w: video.videoWidth, h: video.videoHeight };
  return { w: state.W, h: state.H };
}
function coverRect() {
  const { w, h } = srcSize(), s = Math.max(state.W / w, state.H / h);
  return { x: (state.W - w * s) / 2, y: (state.H - h * s) / 2, w: w * s, h: h * s };
}
function toScreen(p) {           // normalised, NOT mirrored -> screen px
  const r = coverRect(), nx = state.mirror ? 1 - p.x : p.x;
  return { x: r.x + nx * r.w, y: r.y + p.y * r.h };
}
function screenToSource(sx, sy) { // screen px -> normalised source coords
  const r = coverRect();
  let nx = (sx - r.x) / r.w;
  if (state.mirror) nx = 1 - nx;
  return { x: nx, y: (sy - r.y) / r.h };
}

// ---------------------------------------------------------------------------
// MockTracker for touch test mode: same detect() shape as tracker.js.
// Finger position = palm, finger down = fist. Builds a simple fake 21-point
// hand so the skeleton drawing and tips are exercised too.
// ---------------------------------------------------------------------------
const MOCK_FINGERS = [ // base (hand units), direction, segment length; thumb..pinky
  { b: [-0.3, 0.55], d: [-0.75, -0.65], l: 0.32 },
  { b: [-0.33, -0.45], d: [-0.15, -1], l: 0.3 },
  { b: [-0.1, -0.52], d: [0, -1], l: 0.33 },
  { b: [0.14, -0.48], d: [0.12, -1], l: 0.3 },
  { b: [0.36, -0.38], d: [0.3, -1], l: 0.24 },
];
function createMockTracker() {
  const m = { x: 0.5, y: 0.5, down: false, present: true, forced: null };
  m.detect = () => {
    if (!m.present) return null;
    const gesture = m.forced || (m.down ? "fist" : "open");
    const closure = gesture === "fist" ? 1 : gesture === "scissors" ? 0.5 : 0;
    const fingers = gesture === "fist" ? [false, false, false, false, false]
      : gesture === "scissors" ? [false, true, true, false, false] : [true, true, true, true, true];
    const { w, h } = srcSize(), u = 0.09, uy = u * w / h;   // hand unit in normalised x / y
    const pt = (dx, dy) => ({ x: m.x + dx * u, y: m.y + dy * uy, z: 0 });
    const landmarks = [pt(0, 0.9)];
    MOCK_FINGERS.forEach((f, i) => {
      const n = Math.hypot(f.d[0], f.d[1]), len = f.l * (fingers[i] ? 1 : 0.25);
      for (let k = 0; k < 4; k++) landmarks.push(pt(f.b[0] + f.d[0] / n * len * k, f.b[1] + f.d[1] / n * len * k));
    });
    return { palm: { x: m.x, y: m.y }, tips: [4, 8, 12, 16, 20].map(i => ({ x: landmarks[i].x, y: landmarks[i].y })),
             landmarks, size: u, closure, fingers, gesture, score: 1 };
  };
  m.close = () => {};
  return m;
}

// Touch / mouse input for the mock
function mockFromEvent(e) {
  const p = screenToSource(e.clientX, e.clientY);
  state.mock.x = clamp(p.x, 0, 1); state.mock.y = clamp(p.y, 0, 1); state.mock.present = true;
}
cv.addEventListener("pointerdown", e => {
  if (!state.mock || state.useCamera) return;
  mockFromEvent(e); state.mock.down = true; e.preventDefault();
});
window.addEventListener("pointermove", e => {
  if (state.mock && !state.useCamera && (state.mock.down || e.pointerType === "mouse")) mockFromEvent(e);
});
for (const ev of ["pointerup", "pointercancel"]) window.addEventListener(ev, () => { if (state.mock) state.mock.down = false; });

// ---------------------------------------------------------------------------
// Camera + tracker loading
// ---------------------------------------------------------------------------
function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach(t => t.stop());
  state.stream = null; video.srcObject = null; state.res = null;
}

async function openCamera() {
  stopCamera();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const e = new Error("This browser can't use the camera here. Open the page over https in Safari or Chrome.");
    e.name = "NoMediaDevices"; throw e;
  }
  const v = { facingMode: state.facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: v }); }
  catch (e) {
    if (e && e.name === "OverconstrainedError") stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    else throw e;
  }
  state.stream = stream;
  video.srcObject = stream;
  try { await video.play(); } catch (e) { /* autoplay hiccup: frames still arrive on most browsers */ }
  // Mirror what the user sees when it is a front ("user") camera.
  const tr = stream.getVideoTracks()[0];
  const set = tr && tr.getSettings ? tr.getSettings() : {};
  state.mirror = set.facingMode ? set.facingMode !== "environment" : state.facing === "user";
  state.lastVideoTime = -1;
}

function cameraErrorText(e) {
  switch (e && e.name) {
    case "NotAllowedError": case "SecurityError":
      return "Camera permission was denied. Allow camera access for this site in your browser settings and try again, or use Touch test mode.";
    case "NotFoundError": case "OverconstrainedError": case "DevicesNotFoundError":
      return "No camera was found on this device.";
    case "NotReadableError": case "TrackStartError": case "AbortError":
      return "The camera is busy (another app or tab may be using it). Close it and try again.";
  }
  return (e && e.message) || String(e);
}

function setStatus(t) { $("status").textContent = t || ""; }

async function loadTracker() {
  if (state.tracker) return state.tracker;
  if (!state.trackerPromise) {
    state.trackerPromise = (async () => {
      setStatus("Loading hand tracker…");
      const mod = await import("./tracker.js");
      return mod.createTracker({ onStatus: t => { if (state.screen === "start") setStatus(t); } });
    })();
  }
  try { state.tracker = await state.trackerPromise; }
  catch (e) { state.trackerPromise = null; throw e; }
  return state.tracker;
}

let starting = false;
async function startCamera() {
  if (starting) return;
  starting = true;
  const btn = $("btnCam"); btn.disabled = true;
  $("startErr").textContent = ""; setStatus("Starting camera…");
  const camP = openCamera();       // called first, inside the click gesture
  const trkP = loadTracker();
  trkP.catch(() => {});            // handled below; avoid unhandled-rejection noise
  try {
    try { await camP; }
    catch (e) { setStatus(""); $("startErr").textContent = cameraErrorText(e); return; }
    if (!state.tracker) setStatus("Loading hand model…");
    try { await trkP; }
    catch (e) {
      stopCamera(); setStatus("");
      $("startErr").textContent = "The hand-tracking model could not be loaded" +
        (e && e.message ? ": " + e.message : ".") + " Check your internet connection and try again, or use Touch test mode.";
      return;
    }
    setStatus("");
    state.useCamera = true; state.mock = null;
    goMenu();
  } finally { starting = false; btn.disabled = false; }
}

function startTouch() {
  stopCamera();
  state.useCamera = false; state.mirror = false;
  state.mock = createMockTracker();
  state.mock.x = 0.5; state.mock.y = 0.5;
  goMenu();
}

// ---------------------------------------------------------------------------
// Hand detection (one call per new video frame) -> state.hand in screen px
// ---------------------------------------------------------------------------
function detectHand(now) {
  let res = null;
  if (state.useCamera) {
    if (state.tracker && state.stream && video.readyState >= 2 && video.videoWidth) {
      if (video.currentTime !== state.lastVideoTime) {
        state.lastVideoTime = video.currentTime;
        const t = Math.max(now, state.lastDetect + 1);   // tracker needs strictly increasing timestamps
        state.lastDetect = t;
        try { res = state.tracker.detect(video, t) || null; }
        catch (e) { if (!state.detectErr) console.warn("detect() failed", e); state.detectErr = e; res = null; }
      } else res = state.res;                              // same frame: reuse last result
    }
  } else if (state.mock && state.screen !== "start") {
    res = state.mock.detect(null, now);
  }
  state.res = res;
  const h = state.hand;
  if (res && res.palm) {
    const p = toScreen(res.palm);
    h.x = clamp(p.x, 0, state.W); h.y = clamp(p.y, 0, state.H);
    h.closure = clamp(+res.closure || 0, 0, 1);
    h.gesture = res.gesture; h.seen = true; h.lastSeen = now;
  } else {
    h.seen = false;   // keep last position / closure
  }
}

function flash(text, ms = 900) { state.flash.text = text; state.flash.until = state.lastNow + ms; }

// ---------------------------------------------------------------------------
// Best scores (localStorage, may be unavailable)
// ---------------------------------------------------------------------------
const bestKey = n => "handrehab2.best.L" + n;
function loadBest(n) {
  try { const v = parseFloat(localStorage.getItem(bestKey(n))); return Number.isFinite(v) ? v : null; }
  catch { return null; }
}
function saveBest(n, value) {
  const lower = LEVELS[n].lowerBetter, old = loadBest(n);
  const isNew = old == null || (lower ? value < old : value > old);
  if (isNew) { try { localStorage.setItem(bestKey(n), String(value)); } catch {} }
  return { best: isNew ? value : old, isNew };
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function drawSkeleton(res) {
  const lm = res.landmarks;
  if (!lm || lm.length < 21) return;
  const pts = lm.map(toScreen);
  g.save();
  g.strokeStyle = "rgba(232,238,244,0.45)"; g.lineWidth = 2; g.lineCap = "round";
  g.beginPath();
  for (const [a, b] of BONES) { g.moveTo(pts[a].x, pts[a].y); g.lineTo(pts[b].x, pts[b].y); }
  g.stroke();
  g.fillStyle = "rgba(74,124,255,0.7)";
  for (const p of pts) { g.beginPath(); g.arc(p.x, p.y, 3, 0, Math.PI * 2); g.fill(); }
  g.restore();
}

function closureColor(c) { // open = light, closed = amber
  const a = [232, 238, 244], b = [255, 179, 71];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * c)).join(",")})`;
}

function drawCursor() {
  const h = state.hand, base = Math.max(22, Math.min(state.W, state.H) * 0.055);
  const grabbing = state.level && state.level.grabbing;
  g.save();
  g.globalAlpha = h.seen ? 1 : 0.35;
  g.lineWidth = 4; g.strokeStyle = grabbing ? "#ffb347" : closureColor(h.closure);
  g.beginPath(); g.arc(h.x, h.y, base * (1 - 0.4 * h.closure), 0, Math.PI * 2); g.stroke();
  g.fillStyle = g.strokeStyle; g.beginPath(); g.arc(h.x, h.y, 3, 0, Math.PI * 2); g.fill();
  g.restore();
}

function text(str, x, y, size, color = "#e8eef4", weight = 700) {
  g.font = `${weight} ${Math.round(size)}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(str, x, y);
}

function ring(x, y, r, frac, color, width, track = "rgba(42,58,74,0.85)") {
  g.lineWidth = width; g.lineCap = "round";
  if (track) { g.strokeStyle = track; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke(); }
  if (frac > 0) {
    g.strokeStyle = color; g.beginPath();
    g.arc(x, y, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * clamp(frac, 0, 1)); g.stroke();
  }
}

// ---------------------------------------------------------------------------
// Level 1: open & close (10 reps), then grip-and-hold (3 x 3 s)
// ---------------------------------------------------------------------------
function makeLevel1() {
  const REPS = 10, HOLDS = 3, HOLD_MS = 3000;
  const L = { id: 1, phase: "reps", reps: 0, closed: false, holds: 0, holding: false, holdStart: 0,
              needOpen: false, t0: 0, phaseT: 0, message: "" };
  L.start = now => { L.t0 = now; };
  L.update = now => {
    const h = state.hand, c = h.closure, seen = h.seen || now - h.lastSeen < 300;
    if (L.phase === "reps") {
      if (!seen) return;
      if (!L.closed && c > 0.7) L.closed = true;
      else if (L.closed && c < 0.3) {
        L.closed = false; L.reps++; flash(L.reps >= REPS ? "Great!" : "+1");
        if (L.reps >= REPS) { L.phase = "break"; L.phaseT = now; }
      }
    } else if (L.phase === "break") {
      if (now - L.phaseT > 1800) { L.phase = "hold"; L.needOpen = c > 0.3; }
    } else if (L.phase === "hold") {
      if (L.needOpen) { if (seen && c < 0.3) L.needOpen = false; return; }
      if (!L.holding) {
        if (seen && c > 0.6) { L.holding = true; L.holdStart = now; }
      } else if (!seen || c < 0.4) {
        L.holding = false; flash("Keep squeezing!");
      } else if (now - L.holdStart >= HOLD_MS) {
        L.holding = false; L.holds++; L.needOpen = true; flash(L.holds >= HOLDS ? "Done!" : "Well held!");
        if (L.holds >= HOLDS) finish(L, Math.round((now - L.t0) / 1000));
      }
    }
  };
  L.hud = now => [L.phase === "reps" ? `Reps ${L.reps}/${REPS}` : `Holds ${L.holds}/${HOLDS}`,
                  `${Math.floor((now - L.t0) / 1000)} s`];
  L.draw = now => {
    const { W, H } = state, x = W / 2, y = H * 0.48, r = Math.min(W, H) * 0.2, c = state.hand.closure;
    if (L.phase === "reps") {
      ring(x, y, r + 18, L.reps / REPS, "#4a7cff", 6);                       // overall progress
      ring(x, y, r, c, L.closed ? "#ffb347" : "#3fb68b", 16);                // live closure
      for (const t of [0.3, 0.7]) {                                          // threshold ticks
        const a = -Math.PI / 2 + Math.PI * 2 * t;
        g.strokeStyle = "#93a4b5"; g.lineWidth = 2; g.beginPath();
        g.moveTo(x + Math.cos(a) * (r - 12), y + Math.sin(a) * (r - 12));
        g.lineTo(x + Math.cos(a) * (r + 12), y + Math.sin(a) * (r + 12)); g.stroke();
      }
      text(`${L.reps}/${REPS}`, x, y, r * 0.55);
      L.message = L.closed ? "Now open your hand wide" : "Close your hand into a fist";
    } else if (L.phase === "break") {
      ring(x, y, r, 1, "#3fb68b", 16);
      text("10/10", x, y, r * 0.55);
      L.message = "Next: make a fist and hold it for 3 seconds";
    } else {
      const frac = L.holding ? (now - L.holdStart) / HOLD_MS : 0;
      ring(x, y, r + 18, L.holds / HOLDS, "#4a7cff", 6);
      ring(x, y, r, frac, "#ffb347", 16);
      text(L.holding ? Math.max(0, (HOLD_MS - (now - L.holdStart)) / 1000).toFixed(1) : `${L.holds + 1}/${HOLDS}`,
           x, y, r * 0.5);
      L.message = L.needOpen ? "Open your hand" : L.holding ? "Hold it!" : "Make a fist and hold it";
    }
  };
  L.results = v => ({ title: "Well done!", text: `${REPS} open-close reps and ${HOLDS} fist holds in ${v} s.` });
  return L;
}

// ---------------------------------------------------------------------------
// Level 2: rock, paper, scissors (3 rounds vs the computer)
// ---------------------------------------------------------------------------
const RPS = ["fist", "open", "scissors"];
const RPS_ICON = { fist: "✊", open: "✋", scissors: "✌️" };
const RPS_NAME = { fist: "Rock", open: "Paper", scissors: "Scissors" };
const BEATS = { fist: "scissors", scissors: "open", open: "fist" };

function makeLevel2() {
  const ROUNDS = 3, CD_MS = 800, ANSWER_MS = 1500, REVEAL_MS = 2400, READY_MS = 1200;
  const L = { id: 2, round: 1, wins: 0, losses: 0, draws: 0, phase: "ready", t: 0,
              cpu: null, player: null, votes: null, outcome: "", retry: false, message: "" };
  L.start = now => { L.t = now; showRps(!state.useCamera); };
  L.update = now => {
    const el = now - L.t, h = state.hand;
    if (L.phase === "ready" && el >= READY_MS) {
      L.phase = "count"; L.t = now;
      if (state.mock) { state.mock.forced = null; markRps(); }   // touch: choose anew each round
    } else if (L.phase === "count" && el >= 3 * CD_MS) {
      L.phase = "answer"; L.t = now; L.votes = { fist: 0, open: 0, scissors: 0 };
      L.cpu = RPS[Math.floor(Math.random() * 3)];
    } else if (L.phase === "answer") {
      if (h.seen && h.gesture in L.votes) L.votes[h.gesture]++;     // majority of frames
      if (el >= ANSWER_MS) {
        const best = RPS.reduce((a, b) => (L.votes[b] > L.votes[a] ? b : a));
        L.player = L.votes[best] > 0 ? best : null;
        L.retry = !L.player;
        if (L.retry) L.outcome = "none";
        else if (L.player === L.cpu) { L.outcome = "draw"; L.draws++; }
        else if (BEATS[L.player] === L.cpu) { L.outcome = "win"; L.wins++; }
        else { L.outcome = "lose"; L.losses++; }
        L.phase = "reveal"; L.t = now;
      }
    } else if (L.phase === "reveal" && el >= REVEAL_MS) {
      if (!L.retry && L.round >= ROUNDS) { finish(L, L.wins); return; }
      if (!L.retry) L.round++;
      L.phase = "ready"; L.t = now;
    }
  };
  L.hud = () => [`Round ${L.round}/${ROUNDS}`, `You ${L.wins} : ${L.losses} CPU`];
  L.draw = now => {
    const { W, H } = state, x = W / 2, y = H * 0.45, el = now - L.t, big = Math.min(W, H) * 0.22;
    L.message = "";
    if (L.phase === "ready") {
      text("Get ready…", x, y - big * 0.4, big * 0.38);
      text("Rock = fist  ·  Paper = open hand  ·  Scissors = two fingers", x, y + big * 0.3,
           Math.max(13, Math.min(18, W / 30)), "#93a4b5", 600);
    } else if (L.phase === "count") {
      const n = 3 - Math.floor(el / CD_MS);
      text(String(Math.max(1, n)), x, y, big * 1.2, "#e8eef4", 800);
    } else if (L.phase === "answer") {
      text("Show!", x, y - big * 0.55, big * 0.6, "#ffb347", 800);
      const live = state.hand.seen && RPS_ICON[state.hand.gesture];
      text(live || "?", x, y + big * 0.35, big * 0.8);
      const bw = Math.min(W * 0.6, 320), frac = 1 - el / ANSWER_MS;
      g.fillStyle = "rgba(42,58,74,0.85)"; g.fillRect(x - bw / 2, y + big, bw, 8);
      g.fillStyle = "#ffb347"; g.fillRect(x - bw / 2, y + big, bw * clamp(frac, 0, 1), 8);
    } else if (L.phase === "reveal") {
      const dx = Math.min(W * 0.25, big * 1.3);
      text("You", x - dx, y - big * 0.85, big * 0.22, "#93a4b5", 600);
      text("CPU", x + dx, y - big * 0.85, big * 0.22, "#93a4b5", 600);
      text(L.player ? RPS_ICON[L.player] : "❔", x - dx, y, big);
      text(RPS_ICON[L.cpu], x + dx, y, big);
      if (L.player) {
        text(RPS_NAME[L.player], x - dx, y + big * 0.7, big * 0.2, "#93a4b5", 600);
        text(RPS_NAME[L.cpu], x + dx, y + big * 0.7, big * 0.2, "#93a4b5", 600);
      }
      const o = { win: ["You win!", "#3fb68b"], lose: ["You lose", "#ff6b6b"], draw: ["Draw", "#93a4b5"],
                  none: ["Didn't see your hand – again!", "#ffb347"] }[L.outcome];
      text(o[0], x, y + big * 1.1, big * (L.outcome === "none" ? 0.2 : 0.32), o[1], 800);
    }
  };
  L.results = v => ({
    title: L.wins > L.losses ? "You beat the computer!" : L.wins < L.losses ? "The computer wins" : "It's a tie",
    text: `Won ${L.wins}, drew ${L.draws}, lost ${L.losses}.`,
  });
  L.stop = () => showRps(false);
  return L;
}

// Touch-mode gesture buttons
function showRps(on) { $("rps").hidden = !on; markRps(); }
function markRps() {
  for (const b of $("rps").querySelectorAll("button")) b.classList.toggle("on", !!state.mock && state.mock.forced === b.dataset.g);
}
for (const b of $("rps").querySelectorAll("button")) {
  b.addEventListener("click", () => { if (state.mock) { state.mock.forced = b.dataset.g; markRps(); } });
}

// ---------------------------------------------------------------------------
// Level 3: Box and Block (port of v1)
// ---------------------------------------------------------------------------
function makeLevel3() {
  const ROUND_MS = 60000, GRAB_GRACE_MS = 400, LOST_DROP_MS = 500;
  const L = { id: 3, cubes: [], held: null, score: 0, wallHits: 0, endAt: 0,
              grabbing: false, wasGrabbing: false, grabAt: -1e9, message: "" };
  L.layout = () => {
    const { W, H, inset } = state;
    const floor = Math.min(H * 0.86, H - inset.b - 44), wallX = W / 2, wallW = Math.max(14, W * 0.025);
    return { floor, wallX, wallW, wallTop: floor - H * 0.36, cube: Math.max(34, Math.min(W, H) * 0.08) };
  };
  const spawn = () => {
    const Lo = L.layout();
    L.cubes.push({ x: Lo.wallX * (0.2 + Math.random() * 0.55), y: Lo.floor - Lo.cube / 2, vy: 0, side: "L" });
  };
  const drop = () => { if (L.held) { L.held.vy = 0; L.held = null; } L.grabAt = -1e9; };
  L.start = now => { for (let i = 0; i < 4; i++) spawn(); L.endAt = now + ROUND_MS; };
  L.update = (now, dt) => {
    const h = state.hand, Lo = L.layout(), ground = Lo.floor - Lo.cube / 2;
    // grab hysteresis: > 0.6 closes, < 0.4 opens
    if (h.seen) {
      if (!L.grabbing && h.closure > 0.6) { L.grabbing = true; L.grabAt = now; }
      else if (L.grabbing && h.closure < 0.4) L.grabbing = false;
    }
    if (L.held) {
      if (!h.seen) {
        if (now - h.lastSeen > LOST_DROP_MS) drop();   // hand lost: cube falls
      } else if (!L.grabbing) {
        drop();                                          // opened the hand
      } else {
        const prevX = L.held.x;
        const nx = h.x, ny = Math.min(h.y, ground);
        const crossed = (prevX - Lo.wallX) * (nx - Lo.wallX) < 0;
        if (crossed && ny + Lo.cube / 2 > Lo.wallTop) {   // hit the wall below its top
          L.wallHits++; flash("Hit the wall – lift higher!"); drop();
        } else { L.held.x = nx; L.held.y = ny; }
      }
    } else if (h.seen && L.grabbing && now - L.grabAt < GRAB_GRACE_MS) {
      // grab the nearest left-side cube under the cursor (shortly after closing the hand)
      let best = null, bd = Infinity;
      for (const cb of L.cubes) {
        if (cb.side !== "L") continue;
        const dx = Math.abs(cb.x - h.x), dy = Math.abs(cb.y - h.y);
        if (dx < Lo.cube * 0.8 && dy < Lo.cube * 0.8 && dx + dy < bd) { bd = dx + dy; best = cb; }
      }
      if (best) { L.held = best; L.grabAt = -1e9; }
    }
    for (const cb of L.cubes) {
      if (cb === L.held) continue;
      if (cb.y < ground) { cb.vy += 2400 * dt; cb.y = Math.min(ground, cb.y + cb.vy * dt); }
      if (cb.y > ground) cb.y = ground;                  // after a resize
      if (cb.y >= ground && cb.side === "L" && cb.x > Lo.wallX) {
        cb.side = "R"; L.score++; flash("+1"); spawn();
      }
      if (cb.y >= ground) {                              // resting cubes stay clear of the wall
        const half = Lo.wallW / 2 + Lo.cube / 2;
        if (cb.side === "L") cb.x = clamp(cb.x, Lo.cube / 2, Lo.wallX - half);
        else cb.x = clamp(cb.x, Lo.wallX + half, state.W - Lo.cube / 2);
      }
    }
    const right = L.cubes.filter(cb => cb.side === "R");
    if (right.length > 12) L.cubes.splice(L.cubes.indexOf(right[0]), 1);
    L.message = "";
    if (now >= L.endAt) finish(L, L.score);
  };
  L.hud = now => [`Cubes ${L.score}`, `${Math.max(0, Math.ceil((L.endAt - now) / 1000))} s`];
  L.draw = () => {
    const { W, H } = state, Lo = L.layout(), s = Lo.cube;
    if (!state.useCamera) { g.fillStyle = "#1b2836"; g.fillRect(0, Lo.floor, W, H - Lo.floor); }
    g.fillStyle = "rgba(63,182,139,0.14)"; g.fillRect(Lo.wallX, 0, W - Lo.wallX, Lo.floor);
    g.fillStyle = "rgba(27,40,54,0.85)"; g.fillRect(0, Lo.floor, W, 3);
    g.fillStyle = "#5b6b7c"; g.fillRect(Lo.wallX - Lo.wallW / 2, Lo.wallTop, Lo.wallW, Lo.floor - Lo.wallTop);
    text("pick up", Lo.wallX / 2, Lo.floor + 22, 14, "#93a4b5", 600);
    text("drop here", Lo.wallX * 1.5, Lo.floor + 22, 14, "#93a4b5", 600);
    for (const cb of L.cubes) {
      g.fillStyle = cb.side === "R" ? "#3fb68b" : "#f2a541";
      g.fillRect(cb.x - s / 2, cb.y - s / 2, s, s);
      g.strokeStyle = cb === L.held ? "#ffffff" : "rgba(0,0,0,0.35)"; g.lineWidth = 2;
      g.strokeRect(cb.x - s / 2, cb.y - s / 2, s, s);
    }
  };
  L.results = v => ({ title: "Time!",
    text: `You moved ${v} cube${v === 1 ? "" : "s"} over the wall` +
          (L.wallHits ? ` (${L.wallHits} wall hit${L.wallHits === 1 ? "" : "s"}).` : ".") });
  return L;
}

const LEVELS = {
  1: { make: makeLevel1, lowerBetter: true,  fmt: v => `Best time: ${v} s` },
  2: { make: makeLevel2, lowerBetter: false, fmt: v => `Best: ${v} win${v === 1 ? "" : "s"}` },
  3: { make: makeLevel3, lowerBetter: false, fmt: v => `Best: ${v} cube${v === 1 ? "" : "s"}` },
};

// ---------------------------------------------------------------------------
// Screens
// ---------------------------------------------------------------------------
function show(id) { for (const s of document.querySelectorAll(".screen")) s.hidden = s.id !== id; }

function endLevel() {
  if (state.level && state.level.stop) state.level.stop();
  state.level = null; $("hud").hidden = true; $("rps").hidden = true;
}

function goMenu() {
  endLevel();
  state.screen = "menu";
  for (const n of [1, 2, 3]) { const b = loadBest(n); $("best" + n).textContent = b == null ? "" : LEVELS[n].fmt(b); }
  $("modeLine").textContent = state.useCamera ? "Camera mode – show your hand to the camera."
    : "Touch test mode – your finger is the hand; touch = fist, lift = open.";
  show("scrMenu");
}

function goStart() {
  endLevel(); stopCamera();
  state.screen = "start"; state.mock = null; state.useCamera = false;
  show("scrStart");
}

function startLevel(n) {
  endLevel();
  state.levelNo = n; state.level = LEVELS[n].make(); state.level.started = false;
  state.screen = "play"; state.flash.until = 0;
  if (state.mock) { state.mock.down = false; state.mock.forced = null; }
  $("hud").hidden = false; show(null);
}

function finish(L, value) {
  if (state.level !== L) return;
  const r = L.results(value), b = saveBest(L.id, value);
  endLevel();
  state.screen = "results";
  $("resTitle").textContent = r.title;
  $("resText").textContent = r.text;
  $("resBest").textContent = (b.isNew ? "New best! " : "") + LEVELS[L.id].fmt(b.best);
  show("scrResults");
}

// ---------------------------------------------------------------------------
// Main loop: step(nowMs) = detect + update + draw. Called from rAF, from a
// watchdog interval when rAF is throttled, or manually via window.__game.step.
// ---------------------------------------------------------------------------
function step(nowMs) {
  let now = typeof nowMs === "number" && Number.isFinite(nowMs) ? nowMs : performance.now();
  if (now < state.lastNow) now = state.lastNow;                 // never run time backwards
  const dt = Math.min(0.05, (now - state.lastNow) / 1000);
  state.lastNow = now;

  detectHand(now);
  const L = state.screen === "play" ? state.level : null;
  if (L) {
    if (!L.started) { L.started = true; L.start(now); }
    L.update(now, dt);
  }
  draw(now);
  updateHud(now);
}

function draw(now) {
  const { W, H, DPR } = state;
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.fillStyle = "#0f1720"; g.fillRect(0, 0, W, H);
  const camOn = state.useCamera && state.stream && video.readyState >= 2 && video.videoWidth;
  if (camOn) {
    const r = coverRect();
    g.save();
    if (state.mirror) { g.translate(W, 0); g.scale(-1, 1); g.drawImage(video, W - r.x - r.w, r.y, r.w, r.h); }
    else g.drawImage(video, r.x, r.y, r.w, r.h);
    g.restore();
    g.fillStyle = "rgba(15,23,32,0.5)"; g.fillRect(0, 0, W, H);  // dim so game objects stand out
  }
  const L = state.screen === "play" ? state.level : null;
  if (L && L.started) L.draw(now);
  if (state.res) drawSkeleton(state.res);
  if (L) drawCursor();
}

function updateHud(now) {
  const L = state.screen === "play" ? state.level : null;
  if (L && L.started) {
    const [a, b] = L.hud(now);
    $("hudA").textContent = a; $("hudB").textContent = b;
  }
  let msg = "";
  if (now < state.flash.until) msg = state.flash.text;
  else if (state.useCamera && !state.hand.seen && (state.screen === "play" || state.screen === "menu"))
    msg = "Show your hand to the camera";
  else if (L) msg = L.message || "";
  const big = $("big");
  if (big.textContent !== msg) big.textContent = msg;
}

let lastRaf = 0;
function frame() {
  lastRaf = performance.now();
  if (state.auto) step(lastRaf);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
// Fallback when rAF is throttled/paused (e.g. hidden or embedded pane).
setInterval(() => { if (state.auto && performance.now() - lastRaf > 250) step(performance.now()); }, 100);

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------
$("btnCam").addEventListener("click", startCamera);
$("btnTouch").addEventListener("click", startTouch);
$("btnFlip").addEventListener("click", () => {
  state.facing = state.facing === "user" ? "environment" : "user";
  $("btnFlip").textContent = state.facing === "user" ? "Camera: front" : "Camera: back";
});
for (const b of document.querySelectorAll("button.level")) b.addEventListener("click", () => startLevel(+b.dataset.level));
$("btnHome").addEventListener("click", goStart);
$("btnQuit").addEventListener("click", goMenu);
$("btnAgain").addEventListener("click", () => startLevel(state.levelNo || 1));
$("btnMenu").addEventListener("click", goMenu);

// Test hook: drive frames manually with __game.step(t) (set __game.state.auto = false first).
window.__game = { step, state, startTouch, startLevel, goMenu, toScreen, screenToSource, createMockTracker };

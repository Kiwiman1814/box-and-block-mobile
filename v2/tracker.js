// tracker.js - hand tracking for the v2 game, using MediaPipe Hand Landmarker.
// Plain ES module, no build step. See CONTRACT.md for the API.
//
//   import { createTracker } from './tracker.js';
//   const tracker = await createTracker({ onStatus: t => console.log(t) });
//   const hand = tracker.detect(videoEl, performance.now()); // HandResult | null
//   tracker.close();

const MP_VERSION = '0.10.35';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Landmark indices
const WRIST = 0;
const PALM_IDS = [0, 5, 9, 13, 17];
const TIP_IDS = [4, 8, 12, 16, 20];
// index..pinky: [MCP, PIP, DIP, TIP]
const FINGER_CHAINS = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

const LOST_RESET_MS = 300;      // reset smoothing after the hand is gone this long
const GESTURE_HOLD_MS = 90;     // a new gesture must persist this long before it is reported

// ---------------------------------------------------------------- utilities

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerpMap = (v, a, b) => clamp01((v - a) / (b - a)); // a -> 0, b -> 1

/** One-Euro filter (Casiez et al.). t in seconds. */
class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }
  reset() {
    this.x = null;
    this.dx = 0;
    this.t = null;
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(value, t) {
    if (this.x === null || this.t === null) {
      this.x = value;
      this.dx = 0;
      this.t = t;
      return value;
    }
    const dt = Math.max(1e-3, t - this.t);
    this.t = t;
    const rawDx = (value - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (rawDx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
}

/** Source dimensions, or null if the source has no frame yet. */
function sourceSize(src) {
  if (!src) return null;
  if (typeof HTMLVideoElement !== 'undefined' && src instanceof HTMLVideoElement) {
    if (src.readyState < 2) return null;
    return [src.videoWidth, src.videoHeight];
  }
  if (typeof HTMLImageElement !== 'undefined' && src instanceof HTMLImageElement) {
    if (!src.complete) return null;
    return [src.naturalWidth, src.naturalHeight];
  }
  if (typeof HTMLCanvasElement !== 'undefined' && src instanceof HTMLCanvasElement) {
    return [src.width, src.height];
  }
  if (typeof ImageBitmap !== 'undefined' && src instanceof ImageBitmap) {
    return [src.width, src.height];
  }
  if (typeof OffscreenCanvas !== 'undefined' && src instanceof OffscreenCanvas) {
    return [src.width, src.height];
  }
  return null;
}

/** Fetch the model with progress reporting. Returns Uint8Array. */
async function fetchModel(url, onProgress) {
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !res.body.getReader) {
    return new Uint8Array(await res.arrayBuffer());
  }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  let lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) {
      const pct = Math.floor((got / total) * 100);
      if (pct !== lastPct && pct % 5 === 0) {
        lastPct = pct;
        onProgress(pct);
      }
    }
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

// ------------------------------------------------------------ hand geometry

/**
 * Pure geometry from 21 landmarks. Coordinates are converted to an isotropic
 * space in "normalised x units" (y scaled by H/W, z is already in ~x scale).
 */
function analyseHand(lm, aspect /* H / W */) {
  const P = (i) => [lm[i].x, lm[i].y * aspect, lm[i].z];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const len = (v) => Math.hypot(v[0], v[1], v[2]);
  const dist = (i, j) => len(sub(P(i), P(j)));
  const bend = (a, b, c) => {
    // flexion angle at b in degrees: 0 = straight
    const u = sub(P(b), P(a));
    const v = sub(P(c), P(b));
    const lu = len(u);
    const lv = len(v);
    if (lu < 1e-6 || lv < 1e-6) return 0;
    const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv);
    return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
  };

  // 2D size (contract: normalised x units, wrist -> middle MCP)
  const size = Math.hypot(lm[9].x - lm[0].x, (lm[9].y - lm[0].y) * aspect) || 1e-6;
  const size3 = dist(WRIST, 9) || 1e-6;

  // Four fingers: combine joint bend (PIP+DIP) and the tip/MCP wrist-distance ratio.
  // Both are rotation and distance invariant; averaging them makes the result
  // robust when one cue is degraded (e.g. finger pointing at the camera).
  const curls = FINGER_CHAINS.map(([m, p, d, t]) => {
    const angle = bend(m, p, d) + bend(p, d, t);         // ~0-25 straight, ~160+ curled
    const angleCurl = lerpMap(angle, 35, 165);
    const ratio = dist(t, WRIST) / (dist(m, WRIST) || 1e-6); // ~1.9 straight, ~1.0 curled
    const ratioCurl = lerpMap(ratio, 1.8, 1.05);
    return 0.5 * angleCurl + 0.5 * ratioCurl;
  });

  // Thumb: tip distance to index MCP relative to hand size (+ a bit of IP bend).
  const thumbRatio = dist(4, 5) / size3;                  // ~0.9+ open, ~0.3 tucked
  const thumbDistCurl = lerpMap(thumbRatio, 0.85, 0.35);
  const thumbBend = lerpMap(bend(2, 3, 4), 20, 70);
  const thumbCurl = 0.75 * thumbDistCurl + 0.25 * thumbBend;

  return { size, curls: [thumbCurl, ...curls] };
}

// ------------------------------------------------------------------ tracker

export async function createTracker(opts = {}) {
  const status = (t) => {
    try {
      opts.onStatus && opts.onStatus(t);
    } catch (_) {
      /* ignore UI errors */
    }
  };

  // 1. library
  status('Loading hand tracking library…');
  let vision;
  try {
    vision = await import(`${MP_BASE}/vision_bundle.mjs`);
  } catch (e) {
    throw new Error(
      'Could not load the hand tracking library. Check your internet connection and reload the page.'
    );
  }
  const { FilesetResolver, HandLandmarker } = vision;

  // 2. wasm fileset + model (in parallel)
  status('Loading hand model… 0%');
  let fileset;
  let modelBuffer = null;
  try {
    const [fs, mb] = await Promise.all([
      FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`),
      fetchModel(MODEL_URL, (pct) => status(`Loading hand model… ${pct}%`)).catch(() => null),
    ]);
    fileset = fs;
    modelBuffer = mb;
  } catch (e) {
    throw new Error(
      'Could not load the hand tracking engine. Check your internet connection and reload the page.'
    );
  }

  // 3. create the landmarker (GPU first, CPU fallback)
  status('Starting hand tracker…');
  const baseOpts = (delegate) => ({
    baseOptions: modelBuffer
      ? { modelAssetBuffer: modelBuffer, delegate }
      : { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  async function make(delegate) {
    return HandLandmarker.createFromOptions(fileset, baseOpts(delegate));
  }

  let landmarker = null;
  let delegate = 'GPU';
  try {
    landmarker = await make('GPU');
  } catch (eGpu) {
    delegate = 'CPU';
    try {
      landmarker = await make('CPU');
    } catch (eCpu) {
      console.error('[tracker] GPU error:', eGpu, 'CPU error:', eCpu);
      throw new Error(
        'The hand tracker could not start on this device. Try another browser (Chrome or Safari) and reload.'
      );
    }
  }
  status('Hand tracker ready');

  // ---- per-tracker state
  let closed = false;
  let lastTs = -Infinity;
  let lastSeenMs = -Infinity;
  let switching = false; // GPU -> CPU fallback in progress

  // smoothing: palm & tips (x,y) responsive; closure/curls a bit steadier
  const mkXY = () => [new OneEuro(1.5, 8.0, 1.0), new OneEuro(1.5, 8.0, 1.0)];
  const palmF = mkXY();
  const tipF = TIP_IDS.map(mkXY);
  const sizeF = new OneEuro(1.0, 0.5, 1.0);
  const closureF = new OneEuro(2.0, 1.0, 1.0);
  const curlF = [0, 1, 2, 3, 4].map(() => new OneEuro(2.0, 1.0, 1.0));
  const allFilters = [...palmF, ...tipF.flat(), sizeF, closureF, ...curlF];

  let fingersState = [true, true, true, true, true];
  let gesture = 'other';
  let pendingGesture = null;
  let pendingSince = 0;

  function resetState() {
    for (const f of allFilters) f.reset();
    fingersState = [true, true, true, true, true];
    gesture = 'other';
    pendingGesture = null;
  }

  function rawGesture(closure, fingers) {
    // state-dependent thresholds = hysteresis
    const fistOn = gesture === 'fist' ? 0.55 : 0.65;
    const openOn = gesture === 'open' ? 0.38 : 0.3;
    if (closure > fistOn) return 'fist';
    const allExt = fingers.every(Boolean);
    if ((allExt && closure < 0.5) || closure < openOn) return 'open';
    if (fingers[1] && fingers[2] && !fingers[3] && !fingers[4]) return 'scissors';
    return 'other';
  }

  function fallbackToCpu() {
    if (switching || delegate === 'CPU') return;
    switching = true;
    console.warn('[tracker] GPU detection failed, switching to CPU');
    make('CPU')
      .then((lmk) => {
        if (closed) {
          lmk.close();
          return;
        }
        try {
          landmarker && landmarker.close();
        } catch (_) {}
        landmarker = lmk;
        delegate = 'CPU';
        lastTs = -Infinity;
      })
      .catch((e) => console.error('[tracker] CPU fallback failed', e))
      .finally(() => {
        switching = false;
      });
  }

  function detect(source, nowMs) {
    if (closed || switching || !landmarker) return null;
    const dims = sourceSize(source);
    if (!dims || !dims[0] || !dims[1]) return null;
    const [W, H] = dims;

    let ts = typeof nowMs === 'number' && isFinite(nowMs) ? nowMs : performance.now();
    if (ts <= lastTs) ts = lastTs + 1;
    lastTs = ts;

    let res;
    try {
      res = landmarker.detectForVideo(source, ts);
    } catch (e) {
      console.error('[tracker] detectForVideo failed', e);
      if (delegate === 'GPU') fallbackToCpu();
      return null;
    }

    const lm = res && res.landmarks && res.landmarks[0];
    if (!lm || lm.length < 21) {
      return null;
    }

    const freshHand = ts - lastSeenMs > LOST_RESET_MS;
    if (freshHand) resetState();
    lastSeenMs = ts;
    const t = ts / 1000;

    const handed = (res.handedness && res.handedness[0] && res.handedness[0][0]) ||
      (res.handednesses && res.handednesses[0] && res.handednesses[0][0]);
    const score = handed && typeof handed.score === 'number' ? handed.score : 1;

    // palm centre
    let px = 0;
    let py = 0;
    for (const i of PALM_IDS) {
      px += lm[i].x;
      py += lm[i].y;
    }
    px /= PALM_IDS.length;
    py /= PALM_IDS.length;
    const palm = { x: palmF[0].filter(px, t), y: palmF[1].filter(py, t) };

    const tips = TIP_IDS.map((i, k) => ({
      x: tipF[k][0].filter(lm[i].x, t),
      y: tipF[k][1].filter(lm[i].y, t),
    }));

    const geo = analyseHand(lm, H / W);
    const curls = geo.curls.map((c, k) => curlF[k].filter(c, t));

    // extended booleans with hysteresis
    fingersState = curls.map((c, k) => {
      if (c < 0.35) return true;
      if (c > 0.55) return false;
      return fingersState[k];
    });

    // closure: mostly the four fingers
    const fingerMean = (curls[1] + curls[2] + curls[3] + curls[4]) / 4;
    const closure = clamp01(closureF.filter(0.85 * fingerMean + 0.15 * curls[0], t));

    // gesture with threshold hysteresis + short hold time
    const g = rawGesture(closure, fingersState);
    if (freshHand) {
      gesture = g; // first frame after (re)acquiring the hand: no hold
      pendingGesture = null;
    } else if (g === gesture) {
      pendingGesture = null;
    } else {
      if (g !== pendingGesture) {
        pendingGesture = g;
        pendingSince = ts;
      }
      if (ts - pendingSince >= GESTURE_HOLD_MS) {
        gesture = g;
        pendingGesture = null;
      }
    }

    return {
      palm,
      tips,
      landmarks: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
      size: sizeF.filter(geo.size, t),
      closure,
      fingers: fingersState.slice(),
      gesture,
      score,
    };
  }

  function close() {
    closed = true;
    try {
      landmarker && landmarker.close();
    } catch (_) {}
    landmarker = null;
  }

  return {
    detect,
    close,
    get delegate() {
      return delegate;
    },
  };
}

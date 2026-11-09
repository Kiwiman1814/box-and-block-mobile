// tracker-tfjs.js - hand tracking for the v2 game, TensorFlow.js engine.
// Uses @tensorflow-models/hand-pose-detection with runtime 'tfjs' (NOT the
// MediaPipe wasm runtime) on the WebGL backend, CPU backend as fallback.
// Same API and HandResult shape as tracker.js - see CONTRACT.md "Engines (v2.1)".
//
//   import { createTracker } from './tracker-tfjs.js';
//   const tracker = await createTracker({ onStatus: t => console.log(t) });
//   const hand = tracker.detect(videoEl, performance.now()); // HandResult | null
//   tracker.close();
//
// estimateHands() is async, so inference runs in the background on the latest
// frame (at most one in flight) and detect() returns the most recent finished
// result (null if no hand or the result is older than STALE_MS).

const TF_VERSION = '4.22.0';
const HPD_VERSION = '2.0.1';
const CDN = 'https://cdn.jsdelivr.net/npm';
const SCRIPTS = [
  `${CDN}/@tensorflow/tfjs-core@${TF_VERSION}/dist/tf-core.min.js`,
  `${CDN}/@tensorflow/tfjs-converter@${TF_VERSION}/dist/tf-converter.min.js`,
  `${CDN}/@tensorflow/tfjs-backend-webgl@${TF_VERSION}/dist/tf-backend-webgl.min.js`,
  `${CDN}/@tensorflow-models/hand-pose-detection@${HPD_VERSION}/dist/hand-pose-detection.min.js`,
];
const CPU_SCRIPT = `${CDN}/@tensorflow/tfjs-backend-cpu@${TF_VERSION}/dist/tf-backend-cpu.min.js`;
// Model weights: the package defaults (tfhub.dev -> kaggle -> storage.googleapis.com,
// all with Access-Control-Allow-Origin: *), so detectorModelUrl/landmarkModelUrl are not set.
const DEFAULT_MODEL_TYPE = 'full';

const STALE_MS = 300;           // detect() returns null if the newest result is older than this
// Frames are copied into our own canvas (longest side <= this) before inference:
// hand-pose-detection takes the image size from element.width/.height, which is the
// CSS/attribute size for <img>/<video> (0 for a video without width attr), so passing the
// element directly gives wrong keypoint coordinates. A canvas has exact dimensions,
// and downscaling makes the pixel upload cheaper (the models run at 192/224 px anyway).
const MAX_INPUT_DIM = 640;
const MAX_WEBGL_ERRORS = 3;     // consecutive WebGL inference errors before switching to CPU

// Landmark indices (MediaPipe order; hand-pose-detection keypoints use the same order)
const KEYPOINT_NAMES = [
  'wrist',
  'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
  'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
  'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
  'pinky_finger_mcp', 'pinky_finger_pip', 'pinky_finger_dip', 'pinky_finger_tip',
];
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
// bones used to estimate the world(metres) -> image(normalised x) scale for z
const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17],
  [17, 18], [18, 19], [19, 20],
];

const LOST_RESET_MS = 300;      // reset smoothing after the hand is gone this long
const GESTURE_HOLD_MS = 90;     // a new gesture must persist this long before it is reported

// ---------------------------------------------------------------- utilities

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerpMap = (v, a, b) => clamp01((v - a) / (b - a)); // a -> 0, b -> 1

/** One-Euro filter (Casiez et al.). t in seconds. (copied from tracker.js) */
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

/** Source dimensions, or null if the source has no frame yet. (copied from tracker.js) */
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

/** Load a classic (UMD) script once. */
const scriptPromises = new Map();
function loadScript(url) {
  if (scriptPromises.has(url)) return scriptPromises.get(url);
  const p = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.async = false;
    s.crossOrigin = 'anonymous';
    s.onload = () => resolve();
    s.onerror = () => {
      scriptPromises.delete(url);
      s.remove();
      reject(new Error(`failed to load ${url}`));
    };
    document.head.appendChild(s);
  });
  scriptPromises.set(url, p);
  return p;
}

// ------------------------------------------------------------ hand geometry

/**
 * Pure geometry from 21 landmarks (copied from tracker.js). Coordinates are converted
 * to an isotropic space in "normalised x units" (y scaled by H/W, z is already in ~x scale).
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

/**
 * hand-pose-detection hand -> 21 MediaPipe-style landmarks normalised 0..1 in the source.
 * keypoints are source pixels (no z); keypoints3D are world metres around the hand centre.
 * z is derived from keypoints3D: relative to the wrist, scaled to normalised x units
 * (like MediaPipe's image landmarks) using the 2D/3D bone length ratio.
 */
function toLandmarks(hand, W, H) {
  const kp = hand && hand.keypoints;
  if (!kp || kp.length < 21) return null;
  // Order check by name (they follow MediaPipe order; reorder if a future version doesn't).
  let pts = kp;
  if (kp[0].name && kp.some((k, i) => k.name !== KEYPOINT_NAMES[i])) {
    const byName = new Map(kp.map((k) => [k.name, k]));
    pts = KEYPOINT_NAMES.map((n) => byName.get(n));
    if (pts.some((p) => !p)) return null;
  }
  const lm = pts.map((p) => ({ x: p.x / W, y: p.y / H, z: 0 }));

  let k3 = hand.keypoints3D;
  if (k3 && k3.length >= 21) {
    if (k3[0].name && k3.some((k, i) => k.name !== KEYPOINT_NAMES[i])) {
      const byName = new Map(k3.map((k) => [k.name, k]));
      k3 = KEYPOINT_NAMES.map((n) => byName.get(n));
    }
    if (k3.every((p) => p && isFinite(p.z))) {
      const aspect = H / W;
      let l2 = 0;
      let l3 = 0;
      for (const [a, b] of BONES) {
        l2 += Math.hypot(lm[b].x - lm[a].x, (lm[b].y - lm[a].y) * aspect);
        l3 += Math.hypot(k3[b].x - k3[a].x, k3[b].y - k3[a].y);
      }
      if (l3 > 1e-6) {
        const s = l2 / l3;
        const z0 = k3[WRIST].z;
        for (let i = 0; i < 21; i++) lm[i].z = (k3[i].z - z0) * s;
      }
    }
  }
  return lm;
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
  const modelType = opts.modelType === 'lite' ? 'lite' : DEFAULT_MODEL_TYPE;

  // 1. libraries (UMD bundles -> window.tf, window.handPoseDetection), in order
  status('Loading hand tracking library…');
  try {
    for (let i = 0; i < SCRIPTS.length; i++) {
      status(`Loading hand tracking library… ${i + 1}/${SCRIPTS.length}`);
      await loadScript(SCRIPTS[i]);
    }
  } catch (e) {
    console.error('[tracker-tfjs]', e);
    throw new Error(
      'Could not load the hand tracking library. Check your internet connection and reload the page.'
    );
  }
  const tf = window.tf;
  const hpd = window.handPoseDetection;
  if (!tf || !hpd || !hpd.createDetector) {
    throw new Error('The hand tracking library did not load correctly. Reload the page.');
  }

  // 2. backend + model (+ warm-up so shaders compile now, not on the first frame)
  let delegate = 'WebGL';
  let detector = null;

  async function setBackend(name) {
    if (name === 'cpu' && !tf.findBackend('cpu')) await loadScript(CPU_SCRIPT);
    const ok = await tf.setBackend(name);
    if (!ok) throw new Error(`tf.setBackend('${name}') failed`);
    await tf.ready();
  }

  async function makeDetector() {
    const d = await hpd.createDetector(hpd.SupportedModels.MediaPipeHands, {
      runtime: 'tfjs',
      modelType,
      maxHands: 1,
    });
    // warm-up on a small blank canvas (also proves the backend really works)
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    const ctx = c.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, 0, 128, 128);
    }
    await d.estimateHands(c, { flipHorizontal: false, staticImageMode: true });
    return d;
  }

  async function start(backend) {
    status(backend === 'webgl' ? 'Starting graphics engine…' : 'Starting CPU engine (slower)…');
    await setBackend(backend);
    status('Loading hand model…');
    const d = await makeDetector();
    return d;
  }

  try {
    detector = await start('webgl');
    delegate = 'WebGL';
  } catch (eGl) {
    console.warn('[tracker-tfjs] WebGL failed, trying CPU:', eGl);
    try {
      detector = await start('cpu');
      delegate = 'CPU';
    } catch (eCpu) {
      console.error('[tracker-tfjs] WebGL error:', eGl, 'CPU error:', eCpu);
      const msg = String((eCpu && eCpu.message) || eCpu);
      if (/fetch|network|load|404|model/i.test(msg)) {
        throw new Error(
          'Could not load the hand model. Check your internet connection and reload the page.'
        );
      }
      throw new Error(
        'The hand tracker could not start on this device. Try another browser (Chrome or Safari) and reload.'
      );
    }
  }
  status('Hand tracker ready');

  // ---- per-tracker state
  let closed = false;
  let busy = false;           // an inference is in flight
  let switching = false;      // WebGL -> CPU fallback in progress
  let glErrors = 0;
  let lastTs = -Infinity;     // last frame timestamp handed to the model
  let lastSeenMs = -Infinity;
  let latest = null;          // latest finished HandResult (or null = no hand)
  let latestDoneMs = -Infinity; // caller-clock time at which `latest` became available

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

  // calls = inferences finished, hands = inferences that found a hand,
  // ms = smoothed ms per inference, error = last error text
  const stats = { calls: 0, hands: 0, ms: 0, error: '' };

  /** Build a HandResult from the model output (same maths as tracker.js). */
  function process(hands, W, H, ts) {
    const hand = hands && hands[0];
    const lm = hand ? toLandmarks(hand, W, H) : null;
    if (!lm) return null;
    stats.hands++;

    const freshHand = ts - lastSeenMs > LOST_RESET_MS;
    if (freshHand) resetState();
    lastSeenMs = ts;
    const t = ts / 1000;

    const score = typeof hand.score === 'number' ? hand.score : 1;

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
      gesture = g;
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
      landmarks: lm,
      size: sizeF.filter(geo.size, t),
      closure,
      fingers: fingersState.slice(),
      gesture,
      score,
    };
  }

  function fallbackToCpu(reason) {
    if (switching || delegate === 'CPU' || closed) return;
    switching = true;
    console.warn('[tracker-tfjs] switching to CPU:', reason || 'WebGL inference failed');
    const old = detector;
    detector = null;
    try {
      old && old.dispose();
    } catch (_) {}
    start('cpu')
      .then((d) => {
        if (closed) {
          try {
            d.dispose();
          } catch (_) {}
          return;
        }
        detector = d;
        delegate = 'CPU';
        glErrors = 0;
      })
      .catch((e) => {
        console.error('[tracker-tfjs] CPU fallback failed', e);
        stats.error = 'CPU fallback failed: ' + String((e && e.message) || e).slice(0, 60);
      })
      .finally(() => {
        switching = false;
      });
  }

  const frameCanvas = document.createElement('canvas');
  const frameCtx = frameCanvas.getContext('2d');

  function startInference(source, srcW, srcH, ts) {
    busy = true;
    const det = detector;
    const t0 = performance.now();
    // keypoints come back in frameCanvas pixels; normalising by its size gives 0..1 in the source
    const k = Math.min(1, MAX_INPUT_DIM / Math.max(srcW, srcH));
    const W = Math.max(1, Math.round(srcW * k));
    const H = Math.max(1, Math.round(srcH * k));
    let p;
    try {
      if (frameCanvas.width !== W) frameCanvas.width = W;
      if (frameCanvas.height !== H) frameCanvas.height = H;
      frameCtx.drawImage(source, 0, 0, W, H);
      // estimateHands reads the canvas synchronously (tf.browser.fromPixels) before its first await.
      p = det.estimateHands(frameCanvas, { flipHorizontal: false, staticImageMode: false });
    } catch (e) {
      p = Promise.reject(e);
    }
    p.then(
      (hands) => {
        if (closed || det !== detector) return;
        const dt = performance.now() - t0;
        stats.calls++;
        stats.ms = stats.calls === 1 ? dt : 0.8 * stats.ms + 0.2 * dt;
        glErrors = 0;
        latest = process(hands, W, H, ts);
        latestDoneMs = ts + dt; // the caller's clock is performance.now() too
      },
      (e) => {
        if (closed) return;
        console.error('[tracker-tfjs] estimateHands failed', e);
        stats.error = String((e && e.message) || e).slice(0, 80);
        latest = null;
        if (delegate === 'WebGL' && ++glErrors >= MAX_WEBGL_ERRORS) fallbackToCpu(stats.error);
      }
    ).finally(() => {
      busy = false;
    });
  }

  function detect(source, nowMs) {
    if (closed) return null;
    let now = typeof nowMs === 'number' && isFinite(nowMs) ? nowMs : performance.now();

    if (!busy && !switching && detector) {
      const dims = sourceSize(source);
      if (dims && dims[0] && dims[1]) {
        let ts = now;
        if (ts <= lastTs) ts = lastTs + 1;
        lastTs = ts;
        startInference(source, dims[0], dims[1], ts);
      }
    }

    if (!latest) return null;
    // ~300 ms; on a slow backend (CPU: 1-2 s per inference) allow 1.5 inference times
    // so the hand doesn't blink out between results.
    if (now - latestDoneMs > Math.max(STALE_MS, 1.5 * stats.ms)) return null;
    return latest;
  }

  function close() {
    closed = true;
    latest = null;
    try {
      detector && detector.dispose();
    } catch (_) {}
    detector = null;
  }

  return {
    detect,
    close,
    stats,
    kind: 'tfjs',
    /** Switch to the CPU backend (slow, but avoids broken GPU drivers). */
    useCpu(reason) {
      fallbackToCpu(reason || 'requested');
    },
    get switching() {
      return switching;
    },
    get delegate() {
      return delegate;
    },
    get modelType() {
      return modelType;
    },
  };
}

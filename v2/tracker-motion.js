// tracker-motion.js - model-free backup hand tracker for the v2 game.
// Plain ES module, no build step, no downloads. See CONTRACT.md "Engines (v2.1)".
//
//   import { createTracker } from './tracker-motion.js';
//   const tracker = await createTracker({ onStatus: t => console.log(t) });
//   const hand = tracker.detect(videoEl, performance.now()); // HandResult | null
//
// How it works (per frame, ~160 px wide grayscale, ~1 ms of JS on a desktop):
//   1. Downscale: the video is drawn at 2x the working size and 2x2-averaged
//      to luminance (cheap anti-aliasing), then a 3x3 box blur in the core.
//   2. Reference: the current frame is compared with a frame ~90 ms old (kept in
//      a small ring), so even a slow hand (1 px/frame) has moved enough to show.
//      When the hand moves fast, the previous frame is used instead (less lag,
//      smaller trailing "ghost").
//   3. Exposure: the reference is gain-compensated (mean ratio). A sudden
//      brightness jump (> 20%) is a global change: the frame is skipped.
//   4. Camera shake: the frame is split into blocks; each textured block finds its
//      best shift (+-4 px search at half resolution, +-1 refinement at full, so
//      up to +-9 px); the MEDIAN of all blocks is the camera motion. A hand only
//      covers a few blocks, so it cannot drag the median. Flat blocks are skipped
//      (shake is invisible there anyway). No majority / shift at the search limit
//      = shake we cannot compensate: the frame is skipped.
//   5. Difference: a pixel "moved" only if it lies OUTSIDE the [min,max] of the
//      shifted reference's 3x3 neighbourhood by more than a noise-adaptive
//      threshold (>= 12). This absorbs the sub-pixel residual of shake/rotation
//      and sensor noise, while a hand that moved a few pixels gives strong edges.
//   6. Global change: > 40% of pixels changed, or > 8% changed spread over
//      >= 13 of 16 image regions -> skip the frame (last position is held).
//      (No slow background model: with a hand-held phone it would be smeared by
//      shake; a still hand is handled by the hold in 9 instead.)
//   7. Motion cells (4x4 px) feed a decaying motion-energy map (tau 160 ms). Cells
//      above a threshold are dilated and grouped into connected components. The
//      chosen component maximises mass weighted by closeness to the previous
//      position (track continuity). A new track needs current motion in 2
//      consecutive frames at the same place (acquisition).
//   8. Position = centroid of THIS frame's motion inside the TOP part of the
//      moving region (a square window, height = its width, from its top).
//      Reason: when the whole arm moves, the region runs down the forearm to the
//      bottom of the image and the plain centroid sits on the wrist/forearm; the
//      top window keeps the cursor on the hand (fingers lead, player raises hand).
//   9. If the hand stops (no current motion in the blob) the last position is
//      held for holdMs (default 1500 ms: longer than the game's 800 ms dwell so a
//      still hand can dwell-grab; opts.holdMs to change), then null.
//      One-Euro smoothing on top.
//
// The per-frame core works on a plain luminance buffer and is exported for tests:
//   const core = _createMotionCore(opts);  core.step(lum, w, h, nowMs) -> HandResult|null
//   _motionCore(core, lum, w, h, nowMs) is the same call as a function.

const WORK_W = 160;          // working width in px
const CELL = 4;              // motion cell size in px
const SHIFT_R = 4;           // camera-shift search radius at HALF res (= +-9 px at WORK_W incl. refinement)
const BLOCK_H = 16;          // block size (half res) for the shift estimate
const REF_MS = 90;           // compare with a frame >= this old (slow hands move > 1.5 px in that time)
const FAST_ON = 0.7;         // above this hand speed (widths/s) compare with the previous frame
const FAST_OFF = 0.45;       // ... back to REF_MS below this
const RING = 8;              // max frames kept for the reference
const RING_MAX_MS = 400;     // ... and never older than this
const MIN_THR = 12;          // min luminance difference that counts as motion
const GLOBAL_FRAC = 0.4;     // > this fraction of changed pixels = global change
const SPREAD_FRAC = 0.08;    // > this fraction changed AND ...
const SPREAD_REGIONS = 13;   // ... change in >= this many of 4x4 image regions = uncompensated camera motion
const MAX_GAIN_JUMP = 0.2;   // > 20% mean-brightness jump in one frame = global change
const ENERGY_TAU_MS = 160;   // decay time constant of the motion energy map
const CELL_ON = 0.35;        // energy (in "cells' worth" of motion) for a cell to be on
const ACQ_MASS = 3.0;        // component energy mass needed to start a track
const KEEP_MASS = 1.2;       // ... and to keep it
const ACQ_FRAMES = 2;        // consecutive frames needed to start a track
const MOVE_MASS = 0.6;       // current-frame motion mass needed to update the position
const NEAR = 0.25;           // continuity scale (normalised x units)
const HOLD_MS = 1500;        // keep the last position this long without motion

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** One-Euro filter (Casiez et al.). t in seconds. (Same as tracker.js.) */
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

function median(arr, n) {
  const a = Array.prototype.slice.call(arr, 0, n).sort((p, q) => p - q);
  return n % 2 ? a[(n - 1) >> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

// ------------------------------------------------------------------- core

/**
 * Per-frame motion core. Works on a plain luminance buffer (Uint8ClampedArray /
 * Uint8Array, w*h, row-major). No DOM. Returns HandResult | null from step().
 */
export function _createMotionCore(opts = {}) {
  const holdMs = opts.holdMs != null ? opts.holdMs : HOLD_MS;

  let w = 0, h = 0, gw = 0, gh = 0, hw = 0, hh = 0;
  let cur, pMax, pMin, tmp, hc, hr, cellM, energy, onMask, dil, labels, stack;
  let ring = [];               // recent blurred frames, newest last: { buf, mean, t }
  let spare = [];              // recycled frame buffers
  let lastMs = -Infinity;

  // tracking state
  let tracking = false;
  let raw = null;              // last raw position {x, y} (normalised)
  let rawSize = 0.15;
  let lastMotionMs = -Infinity;
  let acqCount = 0, acqPos = null;
  let lastScore = 0;
  let fast = false;            // hand moving fast -> short reference lag
  let speed = 0;               // smoothed speed over consecutive moving frames (widths/s)
  let prevStepMs = -Infinity;
  const palmF = [new OneEuro(1.0, 5.0, 1.0), new OneEuro(1.0, 5.0, 1.0)];
  const sizeF = new OneEuro(0.8, 0.5, 1.0);

  const debug = { skipped: '', lagMs: 0, shift: [0, 0], thr: 0, frac: 0, gain: 1, comps: 0, mass: 0, cur: 0 };

  function alloc(nw, nh) {
    w = nw; h = nh;
    gw = Math.ceil(w / CELL); gh = Math.ceil(h / CELL);
    hw = w >> 1; hh = h >> 1;
    const n = w * h, g = gw * gh;
    cur = new Float32Array(n);
    pMax = new Float32Array(n); pMin = new Float32Array(n); tmp = new Float32Array(n);
    hc = new Float32Array(hw * hh); hr = new Float32Array(hw * hh);
    cellM = new Float32Array(g); energy = new Float32Array(g);
    onMask = new Uint8Array(g); dil = new Uint8Array(g);
    labels = new Int32Array(g); stack = new Int32Array(g);
    ring = []; spare = [];
    resetTrack();
  }

  function resetTrack() {
    tracking = false; raw = null; acqCount = 0; acqPos = null; lastScore = 0; fast = false; speed = 0;
    palmF[0].reset(); palmF[1].reset(); sizeF.reset();
    if (energy) energy.fill(0);
  }

  /** Store cur in the ring (cur gets a recycled buffer). */
  function pushFrame(mean, t) {
    ring.push({ buf: cur, mean, t });
    while (ring.length > RING || (ring.length > 1 && t - ring[0].t > RING_MAX_MS)) spare.push(ring.shift().buf);
    cur = spare.pop() || new Float32Array(w * h);
  }

  /** Reference frame: newest one at least REF_MS old (else the oldest one). */
  function pickRef(t) {
    // fast hand: a short reference keeps the lag (and the trailing ghost) small
    const refMs = fast ? 1 : REF_MS;
    for (let k = ring.length - 1; k >= 0; k--) if (t - ring[k].t >= refMs) return ring[k];
    return ring[0];
  }

  /** 3x3 box blur of src (uint8) into dst (float); returns mean. */
  function blurInto(src, dst) {
    for (let y = 0; y < h; y++) {
      const r = y * w;
      for (let x = 0; x < w; x++) {
        tmp[r + x] = src[r + (x > 0 ? x - 1 : 0)] + src[r + x] + src[r + (x < w - 1 ? x + 1 : x)];
      }
    }
    let sum = 0;
    for (let y = 0; y < h; y++) {
      const ra = (y > 0 ? y - 1 : 0) * w, rb = y * w, rc = (y < h - 1 ? y + 1 : y) * w;
      for (let x = 0; x < w; x++) {
        const v = (tmp[ra + x] + tmp[rb + x] + tmp[rc + x]) * (1 / 9);
        dst[rb + x] = v;
        sum += v;
      }
    }
    return sum / (w * h);
  }

  /** 3x3 min and max of ref into pMin / pMax (separable). */
  function minMax3(ref) {
    for (let y = 0; y < h; y++) {
      const r = y * w;
      for (let x = 0; x < w; x++) {
        const a = ref[r + (x > 0 ? x - 1 : 0)], b = ref[r + x], c = ref[r + (x < w - 1 ? x + 1 : x)];
        let mx = a > b ? a : b; if (c > mx) mx = c;
        let mn = a < b ? a : b; if (c < mn) mn = c;
        tmp[r + x] = mx; pMin[r + x] = mn;
      }
    }
    for (let x = 0; x < w; x++) {
      let pa = pMin[x], ta = tmp[x];
      for (let y = 0; y < h; y++) {
        const i = y * w + x;
        const pb = pMin[i], tb = tmp[i];
        const ic = y < h - 1 ? i + w : i;
        const pc = pMin[ic], tc = tmp[ic];
        let mx = ta > tb ? ta : tb; if (tc > mx) mx = tc;
        let mn = pa < pb ? pa : pb; if (pc < mn) mn = pc;
        pMax[i] = mx;
        pMin[i] = mn;
        pa = pb; ta = tb; // keep this row's ORIGINAL values for the next row
      }
    }
  }

  function halfInto(src, dst) {
    for (let y = 0; y < hh; y++) {
      const r0 = 2 * y * w, r1 = r0 + w, o = y * hw;
      for (let x = 0; x < hw; x++) {
        const i = 2 * x;
        dst[o + x] = 0.25 * (src[r0 + i] + src[r0 + i + 1] + src[r1 + i] + src[r1 + i + 1]);
      }
    }
  }

  /** SAD of block [x0,x1)x[y0,y1) between a(x,y) and gain*b(x-sx, y-sy), checkerboard. */
  function sad(a, b, stride, x0, x1, y0, y1, sx, sy, gain, best) {
    let s = 0;
    for (let y = y0; y < y1; y++) {
      const r = y * stride, rp = (y - sy) * stride - sx;
      for (let x = x0 + (y & 1); x < x1; x += 2) {
        const d = a[r + x] - gain * b[rp + x];
        s += d < 0 ? -d : d;
      }
      if (s >= best) return s; // early exit
    }
    return s;
  }

  /**
   * Camera shift (dx, dy) with cur(x,y) ~ ref(x-dx, y-dy), coarse-to-fine:
   * per-block search +-SHIFT_R at half resolution, +-1 refinement at full
   * resolution, then the median of all textured blocks' votes.
   */
  function estimateShift(ref, gain) {
    halfInto(cur, hc); halfInto(ref, hr);
    const R = SHIFT_R, M = R + 1;
    const nbx = Math.max(1, Math.floor((hw - 2 * M) / BLOCK_H));
    const nby = Math.max(1, Math.floor((hh - 2 * M) / BLOCK_H));
    const bw = Math.floor((hw - 2 * M) / nbx), bh = Math.floor((hh - 2 * M) / nby);
    const vx = [], vy = [];
    for (let by = 0; by < nby; by++) {
      for (let bx = 0; bx < nbx; bx++) {
        const x0 = M + bx * bw, y0 = M + by * bh, x1 = x0 + bw, y1 = y0 + bh;
        let tex = 0, cnt = 0;
        for (let y = y0; y < y1 - 1; y++) {
          for (let x = x0 + (y & 1); x < x1 - 1; x += 2) {
            const i = y * hw + x;
            tex += Math.abs(hc[i + 1] - hc[i]) + Math.abs(hc[i + hw] - hc[i]);
            cnt++;
          }
        }
        if (!cnt || tex / cnt < 1.2) continue; // flat block: no information
        let best = Infinity, cx = 0, cy = 0;
        for (let sy = -R; sy <= R; sy++) {
          for (let sx = -R; sx <= R; sx++) {
            const s = sad(hc, hr, hw, x0, x1, y0, y1, sx, sy, gain, best);
            if (s < best || (s === best && sx * sx + sy * sy < cx * cx + cy * cy)) { best = s; cx = sx; cy = sy; }
          }
        }
        // refine at full resolution around (2cx, 2cy)
        best = Infinity;
        let fx = 2 * cx, fy = 2 * cy;
        const X0 = 2 * x0, X1 = 2 * x1, Y0 = 2 * y0, Y1 = 2 * y1, ccx = fx, ccy = fy;
        for (let sy = ccy - 1; sy <= ccy + 1; sy++) {
          for (let sx = ccx - 1; sx <= ccx + 1; sx++) {
            const s = sad(cur, ref, w, X0, X1, Y0, Y1, sx, sy, gain, best);
            if (s < best) { best = s; fx = sx; fy = sy; }
          }
        }
        vx.push(fx); vy.push(fy);
      }
    }
    if (vx.length < 3) return [0, 0];
    const mx = Math.round(median(vx, vx.length)), my = Math.round(median(vy, vy.length));
    let agree = 0;
    for (let k = 0; k < vx.length; k++) if (Math.abs(vx[k] - mx) <= 1 && Math.abs(vy[k] - my) <= 1) agree++;
    // no majority, or the shift is at the edge of the search range: camera moved more
    // than we can compensate -> null (frame is skipped)
    if (agree * 2 < vx.length) return null;
    if (Math.max(Math.abs(mx), Math.abs(my)) >= 2 * SHIFT_R) return null;
    return [mx, my];
  }

  function step(lum, nw, nh, nowMs) {
    if (nw !== w || nh !== h || !cur) alloc(nw, nh);
    const dtMs = isFinite(lastMs) ? Math.min(250, Math.max(1, nowMs - lastMs)) : 33;
    prevStepMs = lastMs;
    lastMs = nowMs;
    debug.skipped = '';

    const mean = blurInto(lum, cur);
    if (!ring.length) {
      pushFrame(mean, nowMs);
      debug.skipped = 'first';
      return held(nowMs);
    }
    const ref = pickRef(nowMs);
    debug.lagMs = nowMs - ref.t;
    const prevMean = ring[ring.length - 1].mean;

    // sudden brightness jump vs the previous frame = exposure change
    if (prevMean > 4 && Math.abs(mean / prevMean - 1) > MAX_GAIN_JUMP) return globalSkip(mean, nowMs, 'exposure');
    const gain = ref.mean > 4 ? Math.min(2, Math.max(0.5, mean / ref.mean)) : 1;
    debug.gain = gain;
    if (Math.abs(gain - 1) > MAX_GAIN_JUMP) return globalSkip(mean, nowMs, 'exposure');

    const sh = estimateShift(ref.buf, gain);
    if (!sh) return globalSkip(mean, nowMs, 'shake');
    const [dx, dy] = sh;
    debug.shift = [dx, dy];
    minMax3(ref.buf);

    // neighbourhood-range difference against the shifted reference
    const m = Math.max(Math.abs(dx), Math.abs(dy)) + 2;
    let total = 0;
    const hist = new Uint32Array(64);
    for (let y = m; y < h - m; y++) {
      const r = y * w, rp = (y - dy) * w - dx;
      for (let x = m; x < w - m; x++) {
        const c = cur[r + x];
        const hi = gain * pMax[rp + x], lo = gain * pMin[rp + x];
        const d = c > hi ? c - hi : lo > c ? lo - c : 0;
        tmp[r + x] = d;
        hist[d > 63 ? 63 : d | 0]++;
        total++;
      }
    }
    // robust noise level from the histogram (median / upper quartile)
    let acc = 0, p50 = 0, p75 = 0;
    for (let k = 0; k < 64; k++) {
      acc += hist[k];
      if (!p50 && acc >= total * 0.5) p50 = k + 0.5;
      if (!p75 && acc >= total * 0.75) { p75 = k + 0.5; break; }
    }
    const thr = Math.min(40, Math.max(MIN_THR, p50 + 4 * (p75 - p50) + 4));
    debug.thr = thr;

    cellM.fill(0);
    const region = new Uint32Array(16), regionN = new Uint32Array(16);
    let changed = 0;
    const cellArea = CELL * CELL;
    for (let y = m; y < h - m; y++) {
      const r = y * w, gr = ((y / CELL) | 0) * gw, ry = ((y * 4) / h) | 0;
      for (let x = m; x < w - m; x++) {
        const reg = ry * 4 + (((x * 4) / w) | 0);
        regionN[reg]++;
        if (tmp[r + x] > thr) {
          changed++;
          region[reg]++;
          cellM[gr + ((x / CELL) | 0)] += 1 / cellArea;
        }
      }
    }
    const frac = changed / Math.max(1, total);
    debug.frac = frac;
    let active = 0;
    for (let k = 0; k < 16; k++) if (region[k] > 0.03 * regionN[k]) active++;
    if (frac > GLOBAL_FRAC) return globalSkip(mean, nowMs, 'global');
    // change spread over (almost) the whole image = camera motion we could not compensate
    if (frac > SPREAD_FRAC && active >= SPREAD_REGIONS) return globalSkip(mean, nowMs, 'spread');

    pushFrame(mean, nowMs);

    // cells with just one or two changed pixels are noise
    for (let i = 0; i < cellM.length; i++) if (cellM[i] < 3 / cellArea) cellM[i] = 0;

    // motion energy (decaying sum)
    const decay = Math.exp(-dtMs / ENERGY_TAU_MS);
    for (let i = 0; i < energy.length; i++) energy[i] = energy[i] * decay + cellM[i];

    return track(nowMs);
  }

  function globalSkip(mean, nowMs, why) {
    // references from before the change are no longer comparable: restart the ring
    for (const f of ring) spare.push(f.buf);
    ring = [];
    pushFrame(mean, nowMs);
    debug.skipped = why;
    for (let i = 0; i < energy.length; i++) energy[i] *= 0.5;
    return held(nowMs);
  }

  /** Find components in the energy map and update the track. */
  function track(nowMs) {
    // on cells + 1-cell dilation (joins finger / edge fragments)
    for (let i = 0; i < energy.length; i++) onMask[i] = energy[i] > CELL_ON ? 1 : 0;
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        let v = 0;
        for (let yy = Math.max(0, y - 1); yy <= Math.min(gh - 1, y + 1) && !v; yy++)
          for (let xx = Math.max(0, x - 1); xx <= Math.min(gw - 1, x + 1); xx++)
            if (onMask[yy * gw + xx]) { v = 1; break; }
        dil[y * gw + x] = v;
      }
    }
    labels.fill(0);
    const aspect = h / w; // y scale into normalised-x units
    let best = null, bestScore = 0, nComp = 0;
    for (let s = 0; s < dil.length; s++) {
      if (!dil[s] || labels[s]) continue;
      nComp++;
      const lab = nComp;
      let sp = 0; stack[sp++] = s; labels[s] = lab;
      let mass = 0, sx = 0, sy = 0, x0 = gw, x1 = -1, y0 = gh, y1 = -1;
      while (sp) {
        const i = stack[--sp];
        const x = i % gw, y = (i / gw) | 0;
        const e = onMask[i] ? energy[i] : 0;
        if (e > 0) {
          mass += e; sx += e * (x + 0.5); sy += e * (y + 0.5);
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
        for (let yy = y - 1; yy <= y + 1; yy++) {
          if (yy < 0 || yy >= gh) continue;
          for (let xx = x - 1; xx <= x + 1; xx++) {
            if (xx < 0 || xx >= gw) continue;
            const j = yy * gw + xx;
            if (dil[j] && !labels[j]) { labels[j] = lab; stack[sp++] = j; }
          }
        }
      }
      if (mass <= 0) continue;
      const cx = sx / mass / gw, cy = sy / mass / gh;
      let sc = mass;
      if (tracking && raw) {
        const d = Math.hypot(cx - raw.x, (cy - raw.y) * aspect);
        sc = mass / (1 + (d / NEAR) * (d / NEAR));
      }
      if (sc > bestScore) { bestScore = sc; best = { lab, mass, cx, cy, x0, x1, y0, y1 }; }
    }
    debug.comps = nComp;
    debug.mass = best ? best.mass : 0;

    const need = tracking ? KEEP_MASS : ACQ_MASS;
    if (!best || best.mass < need) {
      acqCount = 0; acqPos = null;
      return held(nowMs);
    }

    // Extent of THIS frame's motion inside the blob (the energy blob also holds the
    // decaying trail, which would make it too wide).
    let cm0 = 0, ax0 = gw, ax1 = -1, ay0 = gh, ay1 = -1;
    for (let y = best.y0; y <= best.y1; y++) {
      for (let x = best.x0; x <= best.x1; x++) {
        const i = y * gw + x;
        if (labels[i] !== best.lab || !(cellM[i] > 0)) continue;
        cm0 += cellM[i];
        if (x < ax0) ax0 = x; if (x > ax1) ax1 = x; if (y < ay0) ay0 = y; if (y > ay1) ay1 = y;
      }
    }
    if (cm0 < MOVE_MASS) { ax0 = best.x0; ax1 = best.x1; ay0 = best.y0; ay1 = best.y1; }
    // hand window: top part of the moving region, as tall as it is wide (square in px)
    const bwCells = ax1 - ax0 + 1;
    const winY1 = Math.min(ay1, ay0 + Math.max(2, Math.round(bwCells * 1.0)) - 1);
    let cm = 0, cxs = 0, cys = 0, em = 0, exs = 0, eys = 0;
    for (let y = ay0; y <= winY1; y++) {
      for (let x = best.x0; x <= best.x1; x++) {
        const i = y * gw + x;
        if (labels[i] !== best.lab) continue;
        const c = cellM[i];
        if (c > 0) { cm += c; cxs += c * (x + 0.5); cys += c * (y + 0.5); }
        const e = onMask[i] ? energy[i] : 0;
        if (e > 0) { em += e; exs += e * (x + 0.5); eys += e * (y + 0.5); }
      }
    }
    debug.cur = cm;
    const moving = cm >= MOVE_MASS;
    let px, py;
    if (moving) { px = cxs / cm / gw; py = cys / cm / gh; }
    else if (em > 0) { px = exs / em / gw; py = eys / em / gh; }
    else { px = best.cx; py = best.cy; }
    const size = bwCells / gw;

    if (!tracking) {
      // acquisition: same blob, with CURRENT motion, in consecutive frames
      if (!moving) { acqCount = 0; acqPos = null; return null; }
      if (acqPos && Math.hypot(px - acqPos.x, (py - acqPos.y) * aspect) < NEAR) acqCount++;
      else acqCount = 1;
      acqPos = { x: px, y: py };
      if (acqCount < ACQ_FRAMES) return null;
      tracking = true;
      palmF[0].reset(); palmF[1].reset(); sizeF.reset();
      raw = { x: px, y: py };
      lastMotionMs = nowMs;
    } else if (moving) {
      // speed only from consecutive moving frames (a catch-up jump after a pause is not speed)
      if (lastMotionMs === prevStepMs) {
        const v = Math.hypot(px - raw.x, (py - raw.y) * aspect) / Math.max(0.005, (nowMs - lastMotionMs) / 1000);
        speed += 0.5 * (v - speed);
      }
      raw = { x: px, y: py };
      lastMotionMs = nowMs;
    } else if (nowMs - lastMotionMs > holdMs) {
      // only residual (decaying) energy for too long
      resetTrack();
      return null;
    }
    // when not moving the residual energy may drift towards the trail: keep raw.
    if (!moving) speed = 0;
    fast = fast ? speed > FAST_OFF : speed > FAST_ON;
    rawSize = size;
    lastScore = clamp01(best.mass / (3 * ACQ_MASS)) * (moving ? 1 : 0.6);
    return emit(nowMs, lastScore);
  }

  /** No usable measurement this frame: hold the last position for holdMs. */
  function held(nowMs) {
    if (!tracking || !raw) return null;
    speed = 0; fast = false;
    const age = nowMs - lastMotionMs;
    if (age > holdMs) { resetTrack(); return null; }
    return emit(nowMs, lastScore * (1 - 0.7 * (age / holdMs)));
  }

  function emit(nowMs, score) {
    const t = nowMs / 1000;
    const x = clamp01(palmF[0].filter(raw.x, t));
    const y = clamp01(palmF[1].filter(raw.y, t));
    return {
      palm: { x, y },
      tips: [],
      landmarks: null,
      size: Math.max(0.02, sizeF.filter(rawSize, t)),
      closure: 0,
      fingers: [false, false, false, false, false],
      gesture: 'other',
      score: clamp01(score),
      motion: true,
    };
  }

  return {
    step,
    reset() { for (const f of ring) spare.push(f.buf); ring = []; resetTrack(); },
    debug,
    get tracking() { return tracking; },
  };
}

/** Functional alias of core.step() for tests. */
export function _motionCore(core, lum, w, h, nowMs) {
  return core.step(lum, w, h, nowMs);
}

// --------------------------------------------------------------- DOM part

/** Source dimensions, or null if the source has no frame yet. (As tracker.js.) */
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

export async function createTracker(opts = {}) {
  const status = (t) => {
    try {
      opts.onStatus && opts.onStatus(t);
    } catch (_) {
      /* ignore UI errors */
    }
  };

  const core = _createMotionCore(opts);
  const stats = { calls: 0, hands: 0, ms: 0, error: '' };
  let closed = false;
  let lastTs = -Infinity;
  let canvas = null, ctx = null, lum = null;
  let cw = 0, ch = 0;

  function ensureCanvas(W2, H2) {
    if (!canvas) {
      canvas = typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(W2, H2)
        : document.createElement('canvas');
    }
    if (canvas.width !== W2 || canvas.height !== H2 || !ctx) {
      canvas.width = W2; canvas.height = H2;
      ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
      if (ctx) { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'low'; }
    }
    return ctx;
  }

  function detect(source, nowMs) {
    if (closed) return null;
    const dims = sourceSize(source);
    if (!dims || !dims[0] || !dims[1]) return null;
    const [W, H] = dims;

    let ts = typeof nowMs === 'number' && isFinite(nowMs) ? nowMs : performance.now();
    if (ts <= lastTs) ts = lastTs + 1;
    lastTs = ts;

    const t0 = performance.now();
    let res = null;
    try {
      // working size: WORK_W wide (or the source width if smaller), keep aspect
      const w = Math.max(16, Math.min(WORK_W, Math.floor(W / 2)) | 0);
      const h = Math.max(12, Math.round((w * H) / W));
      const c = ensureCanvas(w * 2, h * 2);
      if (!c) throw new Error('2D canvas unavailable');
      c.drawImage(source, 0, 0, w * 2, h * 2);
      const data = c.getImageData(0, 0, w * 2, h * 2).data;
      if (!lum || cw !== w || ch !== h) { lum = new Uint8ClampedArray(w * h); cw = w; ch = h; }
      // 2x2 average -> luminance (BT.601 weights, integer)
      const W2 = w * 2 * 4;
      for (let y = 0; y < h; y++) {
        let i = y * 2 * W2;
        const o = y * w;
        for (let x = 0; x < w; x++, i += 8) {
          const j = i + W2;
          const r = data[i] + data[i + 4] + data[j] + data[j + 4];
          const g = data[i + 1] + data[i + 5] + data[j + 1] + data[j + 5];
          const b = data[i + 2] + data[i + 6] + data[j + 2] + data[j + 6];
          lum[o + x] = (r * 77 + g * 150 + b * 29) >> 10;
        }
      }
      res = core.step(lum, w, h, ts);
      stats.error = '';
    } catch (e) {
      stats.error = String((e && e.message) || e).slice(0, 80);
      res = null;
    }
    stats.calls++;
    stats.ms = 0.8 * stats.ms + 0.2 * (performance.now() - t0);
    if (res) stats.hands++;
    return res;
  }

  function close() {
    closed = true;
    canvas = null; ctx = null; lum = null;
  }

  status('Motion tracking ready');
  return {
    detect,
    close,
    stats,
    kind: 'motion',
    get delegate() {
      return 'Motion';
    },
  };
}

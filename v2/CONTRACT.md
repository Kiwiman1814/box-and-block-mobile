# v2 contract between tracker.js and the game (index.html + game.js)

Static site served by GitHub Pages at https://kiwiman1814.github.io/box-and-block-mobile/v2/
Plain ES modules, no build step. Phones first (iOS Safari + Android Chrome), must also work on desktop Chrome/Edge.

## tracker.js  (owner: tracker agent)

```js
export async function createTracker(opts = {}) // -> Tracker
//   opts.onStatus?: (text: string) => void     // progress text while the model loads ("Loading hand model…")
//   rejects with an Error whose message is user-readable if the model can't load

// Tracker:
tracker.detect(source, nowMs) // -> HandResult | null   (null = no hand in view)
//   source: HTMLVideoElement (also accept HTMLImageElement / HTMLCanvasElement for tests)
//   nowMs: performance.now(); must be strictly increasing between calls on the same tracker
tracker.close()
```

```js
HandResult = {
  palm:  { x, y },            // palm centre, normalised 0..1 in the SOURCE image (NOT mirrored)
  tips:  [{ x, y }, ...5],    // thumb, index, middle, ring, pinky fingertips, normalised, not mirrored
  landmarks: [{ x, y, z }...21], // raw MediaPipe landmarks (normalised, not mirrored)
  size:  number,              // hand scale: distance wrist -> middle-finger knuckle (normalised x units)
  closure: number,            // 0 = fully open .. 1 = tight fist. Works WITHOUT calibration, from joint angles
  fingers: [bool x5],         // extended? thumb..pinky
  gesture: 'open' | 'fist' | 'scissors' | 'other',   // rock = 'fist', paper = 'open'
  score: number               // detection confidence 0..1
}
```

Mirroring is the GAME's job (front camera shown mirrored: x_screen_norm = 1 - x).
The tracker smooths jitter a little (One-Euro or EMA) but must stay responsive.

## Engines (v2.1)

Three interchangeable modules, each `export async function createTracker(opts)` with the API above:

| file | engine | landmarks |
|---|---|---|
| tracker.js | MediaPipe Hand Landmarker (tasks-vision, wasm) | yes, 21 |
| tracker-tfjs.js | TensorFlow.js `@tensorflow-models/hand-pose-detection`, runtime 'tfjs', WebGL backend | yes, 21, same order as MediaPipe |
| tracker-motion.js | no model: frame differencing / motion blob | no |

Every tracker object also has:
- `stats = { calls, hands, ms, error }` – detect() calls, calls that found a hand, smoothed ms per call, last error text ('' if none). Updated in place.
- `delegate` (getter) – short engine label for the status line, e.g. 'GPU', 'CPU', 'WebGL', 'Motion'.
- `kind` – 'mediapipe' | 'tfjs' | 'motion'.

detect() is SYNCHRONOUS (returns HandResult|null). An engine whose model is async (TF.js
estimateHands returns a Promise) must run inference in the background on the latest frame
and have detect() return the most recent finished result (null if it is older than ~300 ms or no hand),
never queueing more than one inference at a time.

For the landmark engines all HandResult fields are as above (landmarks normalised 0..1 in the source,
not mirrored; `size` = 2D dist wrist(0) → middle MCP(9) in normalised x units with y scaled by H/W).

The motion engine returns `{ palm, tips: [], landmarks: null, size, closure: 0, fingers: [false x5],
gesture: 'other', score, motion: true }` where palm = centre of the moving hand region, size = rough
region size (normalised x units). The game grabs by "dwell" (hold still) in motion mode.

Pinch (thumb tip 4 ↔ index tip 8) is computed by the GAME from `landmarks` + `size`, so engines don't need to.

## game (owner: game agent)

- Imports `{ createTracker }` from `./tracker.js`.
- For touch test mode and for development without the real tracker, the game has its own
  `MockTracker` with the same `detect()` shape (finger position = palm, finger down = fist).

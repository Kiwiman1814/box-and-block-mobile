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

## game (owner: game agent)

- Imports `{ createTracker }` from `./tracker.js`.
- For touch test mode and for development without the real tracker, the game has its own
  `MockTracker` with the same `detect()` shape (finger position = palm, finger down = fist).

# Image Processing Demo (Vision Lab)

This demo showcases AGOLOS image processing in the ZenOne web app:
- Face landmarks and bounding box overlays
- ROI sampling (forehead + cheeks)
- Live rPPG vitals (HR) with quality gating
- Motion, FPS, and brightness diagnostics

## Run

1) `cd ZenOne`
2) `npm install`
3) `npm run dev`
4) Open `http://localhost:5173/?demo=vision`

## Notes

- Requires camera permissions in the browser.
- Uses `@tensorflow-models/face-landmarks-detection` + `@tensorflow/tfjs`.
- ROI colors reflect the live sampled skin regions used for rPPG.

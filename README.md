# Gastronomy AI Server

FastAPI backend for the Gastronomy AI delta food detection system.

## Models
Models are hosted as **GitHub Release assets** (too large for the repo):
- segformer.onnx — SegFormer MiT-B4, 104-class food segmentation (234 MB)
- classifier.onnx — EfficientNet, 80-class Indian food classifier (42 MB)

Downloaded automatically at server cold-start from GitHub Releases v1.0.

## Endpoint
`POST /analyze`
- `prev_image` (JPEG): plate before food placement
- `curr_image` (JPEG): plate after food placement
- `prev_weight_g` (float): scale reading before
- `curr_weight_g` (float): scale reading after

Returns: `{ dish, confidence, seg_class, calories_kcal, weight_g }`

## Deployment
Connected to Render via `render.yaml`. Auto-deploys on push to main.

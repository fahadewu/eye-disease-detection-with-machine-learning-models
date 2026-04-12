# Model Weights

Pre-trained Keras `.h5` weights used by the ensemble. All files are tracked with **Git LFS**.

| File | Architecture | Size | Status |
|---|---|---|---|
| `densenet.h5` | DenseNet121 | ~85 MB | Required |
| `mobilenet.h5` | MobileNetV2 | ~30 MB | Required |
| `effnet.h5` | EfficientNetB3 | ~129 MB | Optional |
| `inceptionv3.h5` | InceptionV3 | ~256 MB | Optional |
| `vgg16.h5` | VGG16 | ~170 MB | Optional |

> The ensemble automatically skips any model whose file is missing.
> At least one model file must be present for local inference.

---

## After cloning — get the model files

**Option A — Git LFS (recommended)**
```bash
# Install LFS if you don't have it
brew install git-lfs        # macOS
sudo apt install git-lfs    # Ubuntu/Debian

# Pull LFS objects
git lfs install
git lfs pull
```

**Option B — Minimal download (densenet + mobilenet only)**
```bash
git lfs pull --include="models/densenet.h5,models/mobilenet.h5"
```

---

## No model files? No problem.

The app still runs without local models.
Configure a free **Hugging Face** or **Google Gemini** API key in
**Admin → Settings** to use those APIs as a fallback for every prediction.

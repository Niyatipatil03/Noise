# BSR Noise Detector

Vehicle **Buzz, Squeak & Rattle (BSR)** detection system — fully offline Android APK powered by a CNN14/PANN-style deep learning model.

---

## Architecture Overview

```
Raw Audio (Bluetooth mic / internal mic)
        ↓
Log-Mel Spectrogram (128 mel bins, 16 kHz)
        ↓
CNN14 Backbone  ← 6 ConvBlocks + CBAM Attention
        ↓
Global Avg + Max Pooling
        ↓
Shared Dense(512) + BN + Dropout
        ↓  ↓
Binary Head     Noise-Type Head
OK / NOT OK     IP / Sunroof / Rear / Steering / Door / Glass / Unknown
        ↓
TFLite → Offline Android APK
```

**Model options**:
| Backbone | Params | AUC (expected) | Notes |
|---|---|---|---|
| `cnn14` | ~31 M | ~0.92 | Default, PANN-style with CBAM attention |
| `yamnet_lite` | ~8 M | ~0.89 | Lighter, faster, YAMNet-like front-end |

---

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```
dataset/
├── OK/                    ← your 40 OK audio files (.wav/.mp3/.flac)
└── NOT_OK/
    ├── IP_NOISE/          ← instrument panel BSR
    ├── SUNROOF_NOISE/     ← sunroof / moonroof noise
    ├── REAR_NOISE/        ← rear section noise
    ├── STEERING_NOISE/    ← steering column noise
    ├── DOOR_NOISE/        ← door panel noise
    └── GLASS_NOISE/       ← glass / window noise
```

> If you haven't labelled noise types yet, drop all NOT_OK files into `dataset/NOT_OK/` directly — they'll be tagged `UNKNOWN_NOISE`.

### 3. Train
```bash
# Single train/val split (recommended for 80 samples)
python scripts/train.py --backbone cnn14

# K-fold cross validation
python scripts/train.py --backbone cnn14 --folds 5

# YAMNet-style lighter model
python scripts/train.py --backbone yamnet_lite
```

### 4. Export to TFLite
```bash
python scripts/export_tflite.py --quant float16
```
This creates `models/bsr_detector.tflite`.

### 5. Run prediction on a file
```bash
python scripts/predict.py path/to/audio.wav
```

### 6. Build Android APK
1. Copy `models/bsr_detector.tflite` → `android/app/src/main/assets/`
2. Open `android/` in Android Studio (Hedgehog / Iguana or newer)
3. Build → Generate Signed Bundle/APK → **APK**
4. Install on device — works 100% offline, no internet/WiFi needed

---

## Android App Features
- **Bluetooth mic support** — connects to any Bluetooth SCO headset/mic automatically
- **Built-in mic fallback** — if no Bluetooth device found within 3 s
- **Real-time sliding window** — new 3-second clip every 1 second
- **Live verdict** — green OK / red NOT OK with noise type name
- **Session report** — counts, percentages, noise breakdown
- **Save to file** — plain-text report saved to device storage

---

## Training Tips (small dataset)

- **Data augmentation** is enabled by default (5 copies per sample) — grows 80 → ~480 training samples
- **SpecAugment** masks random time/frequency bands to improve generalisation
- **Focal Loss** helps with hard-to-detect low-intensity BSR sounds
- **Class weights** are computed automatically to handle OK/NOT_OK imbalance
- Use **k-fold cross-validation** (`--folds 5`) for reliable evaluation on 80 samples

---

## Configuration

All parameters live in `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `audio.sample_rate` | 16000 | Hz |
| `audio.clip_duration` | 3.0 | seconds per window |
| `audio.n_mels` | 128 | mel filterbank bins |
| `model.backbone` | `cnn14` | `cnn14` or `yamnet_lite` |
| `augmentation.num_augmented_per_sample` | 5 | augmented copies per training sample |
| `training.epochs` | 100 | max epochs |
| `training.learning_rate` | 1e-4 | Adam LR |

---

## Project Structure

```
Noise/
├── config.yaml
├── requirements.txt
├── dataset/              ← place audio files here
├── src/
│   ├── data/             ← dataset loading & augmentation
│   ├── features/         ← log-mel spectrogram extraction
│   ├── models/           ← CNN14, YAMNet-lite, CBAM attention, losses
│   ├── training/         ← trainer, callbacks
│   └── utils/            ← audio utils, visualisation
├── scripts/
│   ├── train.py          ← main training entry point
│   ├── evaluate.py       ← detailed evaluation + ROC curve
│   ├── predict.py        ← single-file inference
│   └── export_tflite.py  ← TFLite export for Android
├── models/               ← saved Keras + TFLite models
├── outputs/              ← training curves, confusion matrices
└── android/              ← Android Studio project (Kotlin)
    └── app/src/main/
        ├── assets/       ← bsr_detector.tflite goes here
        ├── java/…/       ← NoiseClassifier, AudioRecorder, MainActivity
        └── res/          ← layouts, colors, strings
```

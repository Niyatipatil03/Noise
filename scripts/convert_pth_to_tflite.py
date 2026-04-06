#!/usr/bin/env python3
"""
Convert a PyTorch .pth model → ONNX → TensorFlow SavedModel → TFLite

Pipeline:
    .pth  →  .onnx  →  SavedModel/  →  .tflite

Usage (auto-detect architecture from filename):
    python scripts/convert_pth_to_tflite.py --pth models/my_model.pth

Usage (explicit architecture):
    python scripts/convert_pth_to_tflite.py --pth models/my_model.pth --arch cnn14
    python scripts/convert_pth_to_tflite.py --pth models/my_model.pth --arch yamnet_lite

Options:
    --quant      float32 | float16 | int8   (default: float16)
    --out        output .tflite path        (default: models/bsr_detector.tflite)
    --verify     run a quick sanity-check inference after conversion
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# ── Config ─────────────────────────────────────────────────────────────────────

def load_cfg(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Step 1: Load PyTorch model ─────────────────────────────────────────────────

def load_pytorch_model(pth_path: str, arch: str, cfg: dict):
    """Load the .pth weights into a PyTorch model."""
    try:
        import torch
    except ImportError:
        raise ImportError("Install PyTorch: pip install torch")

    import math
    sr       = cfg["audio"]["sample_rate"]
    clip_len = int(sr * cfg["audio"]["clip_duration"])
    n_mels   = cfg["audio"]["n_mels"]
    T        = math.ceil(clip_len / cfg["audio"]["hop_length"])
    num_types = len(cfg["dataset"]["noise_types"]) + 2   # +OK +UNKNOWN

    if arch == "cnn14":
        from scripts._torch_cnn14 import CNN14_BSR
        model = CNN14_BSR(
            input_shape=(1, n_mels, T),
            num_noise_types=num_types,
            embedding_dim=cfg["model"]["embedding_dim"],
        )
    elif arch == "yamnet_lite":
        from scripts._torch_yamnet_lite import YAMNetLite_BSR
        model = YAMNetLite_BSR(
            input_length=clip_len,
            num_noise_types=num_types,
            embedding_dim=cfg["model"]["embedding_dim"],
        )
    else:
        # Generic: just load the checkpoint as-is (state_dict or full model)
        state = torch.load(pth_path, map_location="cpu")
        if isinstance(state, torch.nn.Module):
            model = state
        elif isinstance(state, dict) and "model" in state:
            model = state["model"]
        else:
            raise ValueError(
                f"Cannot determine model architecture.\n"
                f"Please pass --arch cnn14 or --arch yamnet_lite\n"
                f"or ensure the .pth contains a full nn.Module (not just state_dict)."
            )
        model.eval()
        print(f"[Load] Loaded full model object from {pth_path}")
        return model, (1, n_mels, T)

    # Load weights
    state = torch.load(pth_path, map_location="cpu")
    if isinstance(state, dict):
        # May be state_dict directly or wrapped
        sd = state.get("state_dict", state.get("model_state_dict", state))
        # Strip "module." prefix if saved from DataParallel
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"[Load] Loaded state_dict from {pth_path}")
    else:
        model = state   # full model saved

    model.eval()
    input_shape = (1, n_mels, T)  # (batch, n_mels, T_frames) — CNN14 convention
    print(f"[Load] Model architecture: {arch}")
    print(f"[Load] Input shape: {input_shape}")
    return model, input_shape


# ── Step 2: Export to ONNX ────────────────────────────────────────────────────

def export_onnx(model, input_shape: tuple, onnx_path: str) -> str:
    """Export PyTorch model to ONNX."""
    try:
        import torch
    except ImportError:
        raise ImportError("Install PyTorch: pip install torch")

    dummy = torch.randn(*input_shape, dtype=torch.float32)
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    # Try to figure out output names dynamically
    with torch.no_grad():
        try:
            out = model(dummy)
        except Exception:
            out = None

    output_names = ["binary_out", "type_out"] if (out is not None and isinstance(out, (tuple, list)) and len(out) == 2) else ["output"]

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["spectrogram_input"],
        output_names=output_names,
        dynamic_axes={
            "spectrogram_input": {0: "batch_size"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[ONNX] Exported → {onnx_path}  ({size_mb:.2f} MB)")

    # Validate
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX] Validation PASSED")
    except ImportError:
        print("[ONNX] Skipping validation (onnx not installed)")

    return onnx_path


# ── Step 3: ONNX → TF SavedModel ─────────────────────────────────────────────

def onnx_to_savedmodel(onnx_path: str, saved_model_dir: str) -> str:
    """Convert ONNX to TF SavedModel using onnx2tf."""
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    print(f"[TF] Converting ONNX → SavedModel …")

    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            non_verbose=True,
            output_signaturedefs=True,
        )
    except ImportError:
        # Fallback: use tf2onnx CLI-style via subprocess
        print("[TF] onnx2tf not found, trying tf2onnx backend …")
        _onnx_to_tf_via_tf2onnx(onnx_path, saved_model_dir)

    print(f"[TF] SavedModel saved → {saved_model_dir}")
    return saved_model_dir


def _onnx_to_tf_via_tf2onnx(onnx_path: str, saved_model_dir: str):
    """Fallback: convert via onnx-tensorflow (legacy onnx-tf package)."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(saved_model_dir)
    except ImportError:
        raise ImportError(
            "No ONNX→TF converter found.\n"
            "Install one of:\n"
            "  pip install onnx2tf sng4onnx     ← recommended\n"
            "  pip install onnx-tf               ← legacy alternative"
        )


# ── Step 4: SavedModel → TFLite ───────────────────────────────────────────────

def savedmodel_to_tflite(
    saved_model_dir: str,
    tflite_path: str,
    quant_mode: str = "float16",
    representative_gen=None,
) -> str:
    """Convert TF SavedModel to TFLite."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quant_mode == "float16":
        converter.target_spec.supported_types = [tf.float16]
        print("[TFLite] Quantisation: float16")

    elif quant_mode == "int8":
        if representative_gen is None:
            print("[TFLite] WARNING: INT8 needs representative dataset — falling back to float16")
            converter.target_spec.supported_types = [tf.float16]
        else:
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type  = tf.int8
            converter.inference_output_type = tf.int8
            print("[TFLite] Quantisation: INT8")
    else:
        print("[TFLite] Quantisation: float32 (dynamic range)")

    tflite_model = converter.convert()

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1e6
    print(f"[TFLite] Saved → {tflite_path}  ({size_mb:.2f} MB)")
    return tflite_path


# ── Step 5: Verify TFLite ─────────────────────────────────────────────────────

def verify_tflite(tflite_path: str, input_shape: tuple):
    """Run dummy inference through the TFLite model."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    print(f"\n[Verify] Input  : {in_det[0]['name']}  shape={in_det[0]['shape']}")
    for od in out_det:
        print(f"[Verify] Output : {od['name']}  shape={od['shape']}")

    # Run dummy inference
    dummy = np.random.randn(*in_det[0]["shape"]).astype(np.float32)
    interp.set_tensor(in_det[0]["index"], dummy)
    interp.invoke()

    print("[Verify] Dummy inference outputs:")
    for od in out_det:
        t = interp.get_tensor(od["index"])
        print(f"  {od['name']}: {t.flatten()[:6]}")
    print("[Verify] PASSED ✓")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert .pth → .tflite")
    parser.add_argument("--pth",    required=True, help="Path to .pth file")
    parser.add_argument("--arch",   default="auto",
                        choices=["auto", "cnn14", "yamnet_lite", "generic"],
                        help="Model architecture (default: auto-detect from filename)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quant",  default="float16",
                        choices=["float32", "float16", "int8"])
    parser.add_argument("--out",    default=None,
                        help="Output .tflite path (default: models/bsr_detector.tflite)")
    parser.add_argument("--verify", action="store_true",
                        help="Run sanity-check inference after conversion")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    pth_path    = args.pth
    stem        = Path(pth_path).stem
    onnx_path   = str(Path("models") / f"{stem}.onnx")
    sm_dir      = str(Path("models") / f"{stem}_saved_model")
    tflite_path = args.out or cfg["paths"]["tflite_model"]

    # ── Auto-detect architecture ──────────────────────────────────────────────
    arch = args.arch
    if arch == "auto":
        name_lower = Path(pth_path).name.lower()
        if "cnn14" in name_lower or "pann" in name_lower:
            arch = "cnn14"
        elif "yamnet" in name_lower:
            arch = "yamnet_lite"
        else:
            arch = "generic"
        print(f"[Auto-detect] Architecture: {arch}")

    # ── Run conversion pipeline ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PTH → TFLite Conversion Pipeline")
    print("="*60)
    print(f"  Input  : {pth_path}")
    print(f"  Arch   : {arch}")
    print(f"  Quant  : {args.quant}")
    print(f"  Output : {tflite_path}")
    print("="*60 + "\n")

    # Step 1
    print("── Step 1/4  Load PyTorch model ──")
    model, input_shape = load_pytorch_model(pth_path, arch, cfg)

    # Step 2
    print("\n── Step 2/4  Export to ONNX ──")
    export_onnx(model, input_shape, onnx_path)

    # Step 3
    print("\n── Step 3/4  ONNX → TF SavedModel ──")
    onnx_to_savedmodel(onnx_path, sm_dir)

    # Step 4
    print("\n── Step 4/4  SavedModel → TFLite ──")
    savedmodel_to_tflite(sm_dir, tflite_path, quant_mode=args.quant)

    # Save labels alongside TFLite
    labels_path = str(Path(tflite_path).parent / "labels.json")
    from src.data.bsr_dataset import BINARY_LABELS, NOISE_TYPE_LABELS
    with open(labels_path, "w") as f:
        json.dump({"binary": BINARY_LABELS, "noise_type": NOISE_TYPE_LABELS}, f, indent=2)
    print(f"[Labels] Saved → {labels_path}")

    # Step 5
    if args.verify:
        print("\n── Verifying TFLite model ──")
        verify_tflite(tflite_path, input_shape)

    print("\n" + "="*60)
    print("  Conversion COMPLETE")
    print(f"  TFLite model : {tflite_path}")
    print("="*60)
    print("\nNext step — copy to Android:")
    print(f"  cp {tflite_path} android/app/src/main/assets/bsr_detector.tflite\n")


if __name__ == "__main__":
    main()

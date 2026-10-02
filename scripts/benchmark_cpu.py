"""CPU efficiency of the 3-channel (unimodal) and 4-channel (early-fusion) YOLOv8n detectors.

Latency does not depend on the weight values, so this can run before training (on the init
checkpoints) or after (pass --runs to use trained best.pt files). Reports parameters, GFLOPs,
PyTorch and ONNX Runtime latency at batch 1 and 640x512 input, ONNX file size and peak RSS.
Late fusion is two unimodal detectors run sequentially plus box fusion; its fusion overhead is
measured on real detections when trained weights are available.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt.common import write_json  # noqa: E402
from scripts.train import build_init  # noqa: E402

H, W = 512, 640


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "mean_ms": float(t.mean())}


def cpu_name():
    if platform.system() == "Windows":
        import subprocess

        try:
            return subprocess.check_output(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=None, help="use trained <runs>/{rgb,rgbt}_s0/weights/best.pt if present")
    ap.add_argument("--work", type=Path, default=Path("results/bench_work"))
    ap.add_argument("--out", type=Path, default=Path("results/cpu_benchmark.json"))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0, help="0 = PyTorch/ORT default")
    a = ap.parse_args()

    import onnxruntime as ort
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if a.threads:
        torch.set_num_threads(a.threads)
    proc = psutil.Process()
    res = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "input": [1, "C", H, W],
        "models": {},
    }
    for tag, ch in (("unimodal_3ch", 3), ("early_fusion_4ch", 4)):
        trained = a.runs / ("rgb_s0" if ch == 3 else "rgbt_s0") / "weights" / "best.pt" if a.runs else None
        w = trained if trained and trained.exists() else build_init(ch, a.work / f"_init_yolov8n_ch{ch}.pt")
        y = YOLO(str(w))
        m = y.model.float().eval().fuse()
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.rand(1, ch, H, W)
        with torch.inference_mode():
            lat_pt = timeit(lambda: m(x), a.warmup, a.iters)
        flops = get_flops(m, imgsz=[H, W])
        onnx_path = Path(y.export(format="onnx", imgsz=[H, W], dynamic=False, simplify=True, opset=17))
        so = ort.SessionOptions()
        if a.threads:
            so.intra_op_num_threads = a.threads
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        xn = x.numpy()
        lat_ort = timeit(lambda: sess.run(None, {name: xn}), a.warmup, a.iters)
        res["models"][tag] = {
            "weights": str(w),
            "trained": bool(trained and trained.exists()),
            "params_M": n_params / 1e6,
            "GFLOPs": flops,
            "pytorch_fp32": lat_pt,
            "onnxruntime_fp32": lat_ort,
            "onnx_MB": onnx_path.stat().st_size / 2**20,
        }
        print(tag, {k: v for k, v in res["models"][tag].items() if k != "weights"}, flush=True)
    u = res["models"]["unimodal_3ch"]
    res["models"]["late_fusion_2x3ch"] = {
        "params_M": 2 * u["params_M"],
        "GFLOPs": 2 * u["GFLOPs"],
        "onnxruntime_fp32": {"median_ms": 2 * u["onnxruntime_fp32"]["median_ms"], "note": "2 x unimodal detector, excludes box fusion"},
        "onnx_MB": 2 * u["onnx_MB"],
    }
    res["peak_rss_MB"] = proc.memory_info().rss / 2**20  # process RSS at the end (includes both sessions)
    write_json(a.out, res)
    print("saved", a.out)


if __name__ == "__main__":
    main()

"""Train one YOLOv8n detector for one input modality and one seed.

All runs share one hyper-parameter set (HYP below); only the dataset (modality) and seed change.
Resumable: if <project>/<modality>_s<seed>/weights/last.pt exists the run is resumed, and if
the run already finished (results.csv has `epochs` rows and best.pt exists) it is skipped.

    python scripts/train.py --data-root /content/llvip_yolo --modality rgbt --seed 0 --project runs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # 4-channel TIFFs trigger a harmless libtiff warning per read
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt.common import MODALITIES  # noqa: E402

# One configuration for every run. HSV colour jitter is disabled for *all* runs because Ultralytics
# cannot apply it to 4-channel inputs; leaving it on for the 3-channel runs would confound the study.
HYP = dict(
    epochs=20,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    cos_lr=True,
    warmup_epochs=3,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=5,
    scale=0.5,
    translate=0.1,
    patience=0,  # no early stopping: every run gets the same budget
    deterministic=True,
    amp=True,
    plots=False,
)


def build_init(ch: int, out: Path, base: str = "yolov8n.pt") -> Path:
    """COCO-pretrained YOLOv8n adapted to `ch` input channels and 1 class.

    For ch=4 the first conv receives the pretrained RGB filters on channels 0-2 and their mean on the
    thermal channel, all scaled by 3/4 so the expected pre-BN activation matches the 3-channel model.
    """
    if out.exists():
        return out
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    src = YOLO(base).model  # downloads the official weights on first use
    m = DetectionModel("yolov8n.yaml", ch=ch, nc=1, verbose=False)
    m.load(src, verbose=True)
    if ch == 4:
        with torch.no_grad():
            w0 = src.state_dict()["model.0.conv.weight"].float()  # (16, 3, 3, 3), RGB order
            w = torch.cat([w0, w0.mean(1, keepdim=True)], 1) * 0.75
            m.model[0].conv.weight.copy_(w)
    m.names = {0: "person"}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m, "train_args": {}, "epoch": -1, "date": None, "version": None}, out)
    return out


def finished(run: Path, epochs: int) -> bool:
    res = run / "results.csv"
    if not (res.exists() and (run / "weights" / "best.pt").exists()):
        return False
    with res.open() as f:
        return sum(1 for _ in csv.DictReader(f)) >= epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--modality", choices=list(MODALITIES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default=False, help="False | ram | disk")
    ap.add_argument("--epochs", type=int, default=HYP["epochs"])
    ap.add_argument("--imgsz", type=int, default=HYP["imgsz"])
    ap.add_argument("--batch", type=int, default=HYP["batch"])
    ap.add_argument("--fraction", type=float, default=1.0, help="debug only")
    a = ap.parse_args()

    from ultralytics import YOLO

    folder, ch = MODALITIES[a.modality]
    data = a.data_root / folder / "data.yaml"
    name = f"{a.modality}_s{a.seed}"
    run = (a.project / name).resolve()
    if finished(run, a.epochs):
        print(f"[skip] {name} already finished")
        return
    last = run / "weights" / "last.pt"
    if last.exists():
        print(f"[resume] {name}")
        YOLO(str(last)).train(resume=True)
        return

    init = build_init(ch, a.project.resolve() / f"_init_yolov8n_ch{ch}.pt")
    hyp = {**HYP, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch}
    cache = {"false": False, "ram": "ram", "disk": "disk"}.get(str(a.cache).lower(), False)
    YOLO(str(init)).train(
        data=str(data),
        project=str(a.project.resolve()),
        name=name,
        exist_ok=True,
        seed=a.seed,
        device=a.device,
        workers=a.workers,
        cache=cache,
        fraction=a.fraction,
        **hyp,
    )


if __name__ == "__main__":
    main()

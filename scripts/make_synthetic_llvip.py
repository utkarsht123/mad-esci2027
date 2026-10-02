"""Generate a tiny *synthetic* dataset in the LLVIP folder layout.

For smoke-testing the pipeline only (file formats, channel order, splits, evaluation code).
Numbers produced from this data are meaningless and must never be reported.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-scenes", type=int, default=4)
    ap.add_argument("--per-scene", type=int, default=12)
    ap.add_argument("--test", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for d in ("visible/train", "visible/test", "infrared/train", "infrared/test", "Annotations"):
        (a.out / d).mkdir(parents=True, exist_ok=True)

    def frame(split, stem):
        W, H = 1280, 1024
        bright = rng.uniform(10, 200)  # scene illumination
        rgb = np.clip(rng.normal(bright, 20, (H, W, 3)), 0, 255).astype(np.uint8)
        ir = np.clip(rng.normal(60, 10, (H, W)), 0, 255).astype(np.uint8)
        objs = []
        for _ in range(rng.integers(1, 4)):
            w, h = rng.integers(40, 120), rng.integers(120, 300)
            x, y = rng.integers(0, W - w), rng.integers(0, H - h)
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(rgb, (x, y), (x + w, y + h), color, -1)
            cv2.ellipse(ir, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 230, -1)
            objs.append((x, y, x + w, y + h))
        cv2.imwrite(str(a.out / "visible" / split / f"{stem}.jpg"), rgb)
        cv2.imwrite(str(a.out / "infrared" / split / f"{stem}.jpg"), cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR))
        xml = "<annotation>" + "".join(
            f"<object><name>person</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
            for x1, y1, x2, y2 in objs
        ) + "</annotation>"
        (a.out / "Annotations" / f"{stem}.xml").write_text(xml)

    for s in range(1, a.train_scenes + 1):
        for i in range(a.per_scene):
            frame("train", f"{s:02d}{i:04d}")
    for i in range(a.test):
        frame("test", f"19{i:04d}")
    print("synthetic LLVIP-layout data written to", a.out)


if __name__ == "__main__":
    main()

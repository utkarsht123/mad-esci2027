"""Convert LLVIP into five YOLO datasets that differ *only* in input modality.

Expected LLVIP layout (as distributed by the authors):
    LLVIP/visible/{train,test}/*.jpg
    LLVIP/infrared/{train,test}/*.jpg
    LLVIP/Annotations/*.xml          (Pascal-VOC, class "person", shared by both modalities)

Output (``--out``):
    {rgb,gray,ir,rgbt,rgbt_unpaired}/images/{train,val,test}/<stem>.(png|tiff)
    {...}/labels/{train,val,test}/<stem>.txt       (identical across modalities)
    {...}/data.yaml
    meta.json                                      (split, scene ids, luminance per test image)

Validation split: LLVIP only ships train/test. We hold out whole *scenes* (the first two digits
of the file name) from the training set so that near-duplicate consecutive frames cannot leak
between train and val.
"""
from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt.common import MODALITIES, TIFF_LZW, WORK_H, WORK_W, luminance, to_gray, write_json  # noqa: E402

EXT = {"rgb": "png", "gray": "png", "ir": "png", "rgbt": "tiff", "rgbt_unpaired": "tiff"}


def find_root(root: Path) -> Path:
    for p in [root, *root.rglob("*")]:
        if p.is_dir() and (p / "visible").is_dir() and (p / "infrared").is_dir():
            return p
    raise FileNotFoundError(f"No LLVIP root (with visible/ and infrared/) under {root}")


def parse_voc(xml: Path, sx: float, sy: float) -> list[str]:
    """VOC xml -> YOLO lines at the working resolution."""
    if not xml.exists():
        return []
    r = ET.parse(xml).getroot()
    lines = []
    for o in r.iter("object"):
        if o.findtext("name", "").strip().lower() != "person":
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = np.clip([x1 * sx, x2 * sx], 0, WORK_W)
        y1, y2 = np.clip([y1 * sy, y2 * sy], 0, WORK_H)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(
            f"0 {(x1 + x2) / 2 / WORK_W:.6f} {(y1 + y2) / 2 / WORK_H:.6f} {(x2 - x1) / WORK_W:.6f} {(y2 - y1) / WORK_H:.6f}"
        )
    return lines


def scene_of(stem: str) -> str:
    return stem[:2]


def load_resized(llvip: Path, src_split: str, stem: str):
    rgb = cv2.imread(str(llvip / "visible" / src_split / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(llvip / "infrared" / src_split / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        raise FileNotFoundError(stem)
    h0, w0 = rgb.shape[:2]
    rgb = cv2.resize(rgb, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    ir = cv2.resize(ir, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    return rgb, ir, WORK_W / w0, WORK_H / h0


def process(job):
    llvip, out, src_split, split, stem, partner_stem, mods = job
    unpaired = "rgbt_unpaired" in mods
    rgb, ir, sx, sy = load_resized(llvip, src_split, stem)
    lines = parse_voc(llvip / "Annotations" / f"{stem}.xml", sx, sy)
    arrays = {
        "rgb": rgb,
        "gray": to_gray(rgb),
        "ir": ir,
        "rgbt": np.dstack([rgb[..., ::-1], ir]),  # R,G,B,T
    }
    arrays = {k: v for k, v in arrays.items() if k in mods}
    if partner_stem is not None:  # unpaired thermal: a frame from a different time/scene
        _, ir_other, _, _ = load_resized(llvip, src_split, partner_stem)
        arrays["rgbt_unpaired"] = np.dstack([rgb[..., ::-1], ir_other])
    elif unpaired:
        # test split (and degenerate 1-frame splits): the unpaired model is always *tested* on true pairs
        arrays["rgbt_unpaired"] = np.dstack([rgb[..., ::-1], ir])
    for mod, arr in arrays.items():
        d = out / mod
        img = d / "images" / split / f"{stem}.{EXT[mod]}"
        if not img.exists():
            params = TIFF_LZW if EXT[mod] == "tiff" else [cv2.IMWRITE_PNG_COMPRESSION, 1]
            assert cv2.imwrite(str(img), arr, params), img
        (d / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return stem, split, len(lines), luminance(rgb) if split == "test" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train and N test frames")
    ap.add_argument("--no-unpaired", action="store_true", help="skip the rgbt_unpaired dataset (saves disk)")
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES), help="subset to build (rgb and ir are always built)")
    a = ap.parse_args()

    llvip = find_root(a.llvip)
    stems = {s: sorted(p.stem for p in (llvip / "visible" / s).glob("*.jpg")) for s in ("train", "test")}
    if a.limit:
        stems = {k: v[: a.limit] for k, v in stems.items()}
    print(f"LLVIP at {llvip}: {len(stems['train'])} train, {len(stems['test'])} test frames")

    # scene-level val split
    rng = random.Random(a.seed)
    scenes = sorted({scene_of(s) for s in stems["train"]})
    target = a.val_frac * len(stems["train"])
    val_scenes, n = [], 0
    if len(scenes) > 1:
        for sc in rng.sample(scenes, len(scenes)):
            if n >= target:
                break
            val_scenes.append(sc)
            n += sum(scene_of(s) == sc for s in stems["train"])
        val = [s for s in stems["train"] if scene_of(s) in val_scenes]
    else:  # fallback: contiguous tail block
        val = stems["train"][-int(target):]
    val_set = set(val)
    train = [s for s in stems["train"] if s not in val_set]
    splits = {"train": train, "val": val, "test": stems["test"]}

    want = set(a.modalities) | {"rgb", "ir"}  # evaluation reads the rgb/ir test images for every model
    mods = [m for m in MODALITIES if m in want and not (a.no_unpaired and m == "rgbt_unpaired")]
    for m in mods:
        for sp in splits:
            (a.out / m / "images" / sp).mkdir(parents=True, exist_ok=True)
            (a.out / m / "labels" / sp).mkdir(parents=True, exist_ok=True)
        ch = MODALITIES[m][1]
        (a.out / m / "data.yaml").write_text(
            f"path: {(a.out / m).resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"channels: {ch}\nnames:\n  0: person\n"
        )

    # Unpaired partners: a derangement *within the same split* whose partner comes from a different scene
    # where possible, so the thermal channel carries no information about the RGB frame's people.
    jobs = []
    for sp, lst in splits.items():
        partners = [None] * len(lst)
        if "rgbt_unpaired" in mods and sp in ("train", "val") and len(lst) > 1:
            multi_scene = len({scene_of(t) for t in lst}) > 1
            for i, s in enumerate(lst):
                t = s
                while t == s or (multi_scene and scene_of(t) == scene_of(s)):
                    t = rng.choice(lst)
                partners[i] = t
        src = "test" if sp == "test" else "train"
        jobs += [(llvip, a.out, src, sp, s, p, tuple(mods)) for s, p in zip(lst, partners)]

    lum, n_boxes = {}, {k: 0 for k in splits}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (stem, sp, nb, L) in enumerate(ex.map(process, jobs, chunksize=16)):
            n_boxes[sp] += nb
            if L is not None:
                lum[stem] = L
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    meta = {
        "llvip_root": str(llvip),
        "work_size": [WORK_W, WORK_H],
        "val_scenes": val_scenes,
        "counts": {k: len(v) for k, v in splits.items()},
        "boxes": n_boxes,
        "test_luminance": lum,
        "modalities": mods,
    }
    write_json(a.out / "meta.json", meta)
    print({k: meta[k] for k in ("counts", "boxes", "val_scenes")})


if __name__ == "__main__":
    main()

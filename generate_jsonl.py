import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

ROOT_MUSIC = Path("data/FineDance/music_wav")
ROOT_MOTION = Path("data/FineDance_preprocessed/motion_vec263_fps60")
ROOT_LABEL = Path("data/FineDance/label_json")   # 没有也可以
OUT_DIR = Path("data/FineDance/splits")

VAL_RATIO = 0.1
SEED = 3407
USE_LABEL_JSON = False   # 如果没有 label_json，就改成 False

OUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)


def get_stems(folder: Path, suffix: str) -> set:
    if not folder.exists():
        return set()
    return {p.stem for p in folder.glob(f"*{suffix}") if p.is_file()}


def load_label_meta(stem: str) -> Dict:
    if not USE_LABEL_JSON:
        return {}

    label_path = ROOT_LABEL / f"{stem}.json"
    if not label_path.exists():
        return {}

    with open(label_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "label": f"label_json/{stem}.json",
        "style1": meta.get("style1", "Unknown"),
        "style2": meta.get("style2", "Unknown"),
        "frames": meta.get("frames", None),
    }


def build_item(stem: str) -> Dict:
    item = {
        "id": stem,
        "music": f"{stem}.wav",
        "motion": f"{stem}.npy",
    }
    item.update(load_label_meta(stem))
    return item


def main():
    music_stems = get_stems(ROOT_MUSIC, ".wav")
    motion_stems = get_stems(ROOT_MOTION, ".npy")

    if USE_LABEL_JSON and ROOT_LABEL.exists():
        label_stems = get_stems(ROOT_LABEL, ".json")
        common_stems = sorted(music_stems & motion_stems & label_stems)
    else:
        common_stems = sorted(music_stems & motion_stems)

    print(f"music stems: {len(music_stems)}")
    print(f"motion stems: {len(motion_stems)}")
    if USE_LABEL_JSON and ROOT_LABEL.exists():
        print(f"label stems: {len(label_stems)}")
    print(f"paired stems: {len(common_stems)}")

    if len(common_stems) == 0:
        print("example music stems:", sorted(list(music_stems))[:10])
        print("example motion stems:", sorted(list(motion_stems))[:10])
        raise RuntimeError("No paired samples found. Check your music/motion paths and stem names.")

    # 最简单稳妥版：整体 shuffle，再 9:1 划分
    stems = common_stems[:]
    random.shuffle(stems)

    n_val = max(1, int(round(len(stems) * VAL_RATIO)))
    n_val = min(n_val, len(stems) - 1) if len(stems) > 1 else 0

    val_stems = stems[:n_val]
    train_stems = stems[n_val:]

    train_items: List[Dict] = [build_item(stem) for stem in train_stems]
    val_items: List[Dict] = [build_item(stem) for stem in val_stems]

    train_jsonl = OUT_DIR / "train_pairs.jsonl"
    val_jsonl = OUT_DIR / "val_pairs.jsonl"

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "train.txt", "w", encoding="utf-8") as f:
        for stem in train_stems:
            f.write(stem + "\n")

    with open(OUT_DIR / "val.txt", "w", encoding="utf-8") as f:
        for stem in val_stems:
            f.write(stem + "\n")

    print(f"Train: {len(train_items)}")
    print(f"Val: {len(val_items)}")
    print(f"Wrote: {train_jsonl}")
    print(f"Wrote: {val_jsonl}")


if __name__ == "__main__":
    main()
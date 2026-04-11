from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def compute_bbox(mask_path: Path) -> list[int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {mask_path}")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def convert_sample(item: dict, label_to_id: dict[str, int], data_root: Path) -> dict:
    if len(item["caption"]) <= 3:
        raise ValueError(f"caption[3] missing for sample: {item['id']}")

    return {
        "bbox": compute_bbox(data_root / item["mask"]),
        "cat": label_to_id[item["disease_label"]],
        "segment_id": item["id"],
        "img_name": item["image"],
        "sentences": [
            {
                "idx": 0,
                "sent_id": item["id"],
                "sent": item["caption"][3].strip(),
            }
        ],
        "sentences_num": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PlantSeg metadata into the VLT JSON schema.")
    parser.add_argument("--data-root", default="../plantseg", help="PlantSeg dataset root.")
    parser.add_argument(
        "--output-dir",
        default="data/data_v2/anns/plantseg",
        help="Output directory for converted JSON annotations.",
    )
    parser.add_argument(
        "--label-map",
        default="data/data_v2/anns/plantseg/label_map.json",
        help="Output path for the disease label mapping.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "main.json").read_text(encoding="utf-8"))
    label_to_id = {label: idx for idx, label in enumerate(sorted({item["disease_label"] for item in manifest}))}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.label_map).parent.mkdir(parents=True, exist_ok=True)

    split_to_records: dict[str, list[dict]] = defaultdict(list)
    for item in manifest:
        split = item["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        split_to_records[split].append(convert_sample(item, label_to_id, data_root))

    for split, records in split_to_records.items():
        (output_dir / f"{split}.json").write_text(json.dumps(records), encoding="utf-8")
        print(f"wrote {split}: {len(records)}")

    label_map = {
        "label_to_id": label_to_id,
        "id_to_label": {idx: label for label, idx in label_to_id.items()},
    }
    Path(args.label_map).write_text(json.dumps(label_map, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# Linux 4090 PlantSeg Workflow

This repository now supports a Linux-first single-GPU workflow for PlantSeg.

## Assets

- Keep `plantseg` next to the repository root.
- Run `tools/prepare_server_assets.py` to download the official MCN DarkNet53-YOLOv3 Keras backbone weights into `data/weights/` and install the official `en_core_web_lg` spaCy model.

## Dataset conversion

- Run `tools/prepare_plantseg.py` to convert `../plantseg/main.json` into:
  - `data/data_v2/anns/plantseg/train.json`
  - `data/data_v2/anns/plantseg/val.json`
  - `data/data_v2/anns/plantseg/test.json`
  - `data/data_v2/anns/plantseg/label_map.json`
- The converter uses `caption[3]` as the only text input and keeps the original `train/val/test` split.
- `false_healthy_ann/` is intentionally ignored.

## Environment

- `environment.server.linux.cuda118.yml` is the environment entry point for the server.
- `requirements.server.txt` is the pip dependency lock for the Linux server target.
- The environment file explicitly uses `conda-forge` + `nodefaults` so that stale global `defaults` or mirror entries do not leak into the solve.
- If the server has a custom `.condarc`, prefer `tools/create_server_env.sh`, which disables conda plugins and overrides global channels during environment creation.

## Training

- Training config: `config/plantseg/train.yaml`
- This preserves `vlt.py train` and writes the best checkpoint to `log/plantseg_train/models/best_map.h5`.

## Testing

- Test config: `config/plantseg/test.yaml`
- This preserves `vlt.py test` and evaluates the test split with the final metrics:
  - `IoU`
  - `Dice`
  - `Recall`
  - `mIoU`
  - `mACC`

## Results

- Text results are written under `result/`.
- Structured JSON results are written alongside the text file for downstream parsing.

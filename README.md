# Vision-Language Transformer for Local JSON Referring Segmentation

This repository is adapted for Linux server training and testing on a single RTX 4090 with a local JSON dataset. The original RefCOCO data preparation scripts are kept in the repository for reference, but they are no longer part of the supported workflow.

## Supported workflow

- Train with `vlt.py train <config>`
- Test with `vlt.py test <config>`
- Read data from sibling `../dataset/train.json` and `../dataset/test.json`
- Use one caption per sample, defaulting to `caption[2]`
- Treat every non-zero mask pixel as foreground
- Report only `iou`, `dice`, `recall`, `miou`, `macc` on the held-out test split

## Expected dataset layout

The repository is expected to sit beside the dataset directory:

```text
Segmentation/
  VLT/
  dataset/
    train.json
    test.json
    train/
      img/
      lbl/
    test/
      img/
      lbl/
```

Each JSON item must contain:

```json
{
  "id": "sample_id",
  "image": "train/img/example.jpg",
  "mask": "train/lbl/example.png",
  "caption": [
    "short expression",
    "alternate expression",
    "default expression used by this repo"
  ]
}
```

`image` and `mask` may also be absolute paths.

## Environment

The Linux server environment definition is in `environment.linux.4090.yml`.

Create the environment and then install the spaCy model used by the default config.
The `spacy download` helper can fail in some mirrored or proxied environments, so use the official model wheel directly:

```bash
conda env create -f environment.linux.4090.yml
conda activate vlt-linux-4090
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
```

## External assets

This repository does not track runtime assets in Git.

- Dataset: provide `../dataset`
- YOLO backbone initialization weights: place them under `./weights/` and point `pretrained_weights` at the actual file
- Training outputs: created under `log_path`

The default config expects `./weights/yolov3_480000.h5`. Training will fail fast if it is missing.

## Configuration

The default server-oriented config is `config/base.yaml`.

Key fields:

- `dataset_root`: sibling dataset directory, default `../dataset`
- `train_set`: training JSON file
- `evaluate_set`: held-out test JSON file
- `caption_index`: default `2`, interpreted as a 0-based index
- `word_embed`: default `en_core_web_lg`
- `word_len`: default `32`
- `segment_thresh`: prediction threshold used during testing, default `0.35`
- `pretrained_weights`: backbone init weights for training
- `resume_model`: optional full-model training checkpoint for resume
- `evaluate_model`: checkpoint to test

An example config for the local dataset is provided at `config/local_dataset/example.yaml`.

## Training

Training preserves the original model and learning-rate schedule, with the original default `50` epochs and step decay at `40`, `45`, `50`.

Training no longer runs validation after each epoch. Instead it writes:

- `log_path_<config_name>/models/last.weights.h5`
- `log_path_<config_name>/models/final.weights.h5`

Start training with:

```bash
python vlt.py train config/local_dataset/example.yaml
```

If you want to resume training, set `resume_model` in the config to a previous `last.weights.h5` or compatible full-model checkpoint.

## Testing

Testing uses the held-out JSON split specified by `evaluate_set`. It accumulates global pixel-level:

- TP
- FP
- FN
- TN

and then reports:

- `iou`
- `dice`
- `recall`
- `miou`
- `macc`

If any metric denominator is zero, that metric is recorded as `0`.

Run testing with:

```bash
python vlt.py test config/local_dataset/example.yaml
```

The config should point `evaluate_model` at the checkpoint you want to test, typically `final.weights.h5`.

Results are written to:

- `result/test_metrics.json`
- `result/test_metrics.txt`

## Notes

- The project is now Linux-server-first. Windows local execution is not a supported target.
- The old RefCOCO preparation path under `data/` is retained only as legacy reference.
- The current implementation keeps the repository entry points and YACS config flow intact while adapting the dataset and test logic to the local JSON task.

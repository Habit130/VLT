import json
import os

import cv2
import numpy as np
import progressbar
import spacy

from loader.loader import get_random_data


class Evaluate(object):
    """Standalone evaluator for held-out testing on the local JSON dataset."""

    def __init__(self, model, data, config, verbose=1):
        self.model = model
        self.data = data
        self.config = config
        self.verbose = verbose
        self.batch_size = max(config.batch_size // 2, 1)
        self.input_shape = (config.input_size, config.input_size)
        self.word_embed = spacy.load(config.word_embed)
        self.threshold = config.segment_thresh

    def evaluate(self):
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        iterator = range(0, len(self.data), self.batch_size)
        iterator = progressbar.progressbar(iterator, prefix="evaluation: ")

        for start in iterator:
            batch = self.data[start:start + self.batch_size]
            images = []
            word_vecs = []
            gt_masks = []

            for sample in batch:
                image_data, word_vec, _, _, gt_mask = get_random_data(
                    sample,
                    self.input_shape,
                    self.word_embed,
                    self.config,
                    train_mode=False,
                )
                images.append(image_data)
                word_vecs.append(word_vec)
                gt_masks.append(gt_mask)

            preds = self.model.predict_on_batch([np.array(images), np.array(word_vecs)])
            probs = self.sigmoid(preds)

            for pred, gt_mask in zip(probs, gt_masks):
                pred_mask = self.restore_prediction(pred[:, :, 0], gt_mask.shape[:2])
                pred_bin = pred_mask > self.threshold
                gt_bin = gt_mask[:, :, 0] > 0

                counts["tp"] += int(np.logical_and(pred_bin, gt_bin).sum())
                counts["fp"] += int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
                counts["fn"] += int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
                counts["tn"] += int(np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum())

        metrics = self.compute_metrics(counts)
        if self.verbose:
            print(json.dumps(metrics, indent=2))
        return metrics

    def restore_prediction(self, pred_seg, gt_shape):
        gh, gw = gt_shape
        h, w = self.input_shape

        scale = min(w / gw, h / gh)
        nw = int(gw * scale)
        nh = int(gh * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        pred_seg = cv2.resize(pred_seg, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_seg = pred_seg[dy:dy + nh, dx:dx + nw]
        if pred_seg.size == 0:
            return np.zeros((gh, gw), dtype=np.float32)
        pred_seg = cv2.resize(pred_seg, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return pred_seg

    def compute_metrics(self, counts):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]

        iou = self.safe_div(tp, tp + fp + fn)
        dice = self.safe_div(2 * tp, 2 * tp + fp + fn)
        recall = self.safe_div(tp, tp + fn)
        iou_bg = self.safe_div(tn, tn + fn + fp)
        miou = (iou + iou_bg) / 2.0
        acc_fg = self.safe_div(tp, tp + fn)
        acc_bg = self.safe_div(tn, tn + fp)
        macc = (acc_fg + acc_bg) / 2.0

        return {
            "iou": iou,
            "dice": dice,
            "recall": recall,
            "miou": miou,
            "macc": macc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def save(self, metrics, result_dir, checkpoint_path, dataset_path):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        payload = dict(metrics)
        payload["checkpoint"] = checkpoint_path
        payload["dataset"] = dataset_path

        json_path = os.path.join(result_dir, "test_metrics.json")
        txt_path = os.path.join(result_dir, "test_metrics.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(txt_path, "w", encoding="utf-8") as f:
            for key in ["iou", "dice", "recall", "miou", "macc"]:
                f.write("{}: {:.6f}\n".format(key, metrics[key]))

        return json_path, txt_path

    @staticmethod
    def safe_div(num, denom):
        if denom == 0:
            return 0.0
        return float(num) / float(denom)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

import os

import numpy as np
from loader.loader import get_random_data
import cv2
from matplotlib.pyplot import cm
import progressbar
from tensorflow import keras

from runtime_utils import load_spacy_model

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        data,
        config,
        tensorboard=None,
        verbose=1,
        phase='train'
    ):
        self.val_data = data
        self.tensorboard = tensorboard
        self.verbose = verbose
        self.vis_id = [i for i in np.random.randint(0, len(data), 200)]
        self.batch_size = max(config.batch_size//2, 1)
        self.colors = np.array(cm.hsv(np.linspace(0, 1, 10)).tolist()) * 255
        self.input_shape = (config.input_size, config.input_size)  # multiple of 32, hw
        self.config = config
        self.word_embed = load_spacy_model(config.word_embed)
        self.seg_min_overlap = config.segment_thresh
        if phase == 'test':
            self.log_images = config.log_images
            self.multi_thres = config.multi_thres
        else:
            self.log_images = 0
            self.multi_thres = False
        self.eval_save_images_id = [i for i in np.random.randint(0, len(self.val_data), 200)]
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # run evaluation
        self.seg_metrics = self.evaluate()

        logs['seg_iou'] = self.seg_metrics['IoU']
        logs['seg_dice'] = self.seg_metrics['Dice']
        logs['seg_recall'] = self.seg_metrics['Recall']
        logs['seg_miou'] = self.seg_metrics['mIoU']
        logs['seg_macc'] = self.seg_metrics['mACC']

        if self.verbose == 1:
            print('IoU: {:.4f}, Dice: {:.4f}, Recall: {:.4f}, mIoU: {:.4f}, mACC: {:.4f}'.format(
                self.seg_metrics['IoU'],
                self.seg_metrics['Dice'],
                self.seg_metrics['Recall'],
                self.seg_metrics['mIoU'],
                self.seg_metrics['mACC'],
            ))

    def evaluate(self):
        img_id = 0
        confusion = dict(tp=0, fp=0, fn=0, tn=0)

        test_batch_size = self.batch_size
        for start in progressbar.progressbar(range(0, len(self.val_data), test_batch_size), prefix='evaluation: '):
            end = start + test_batch_size
            batch_data = self.val_data[start:end]
            images = []
            images_ori = []
            files_id = []
            word_vecs = []
            sentences = []
            gt_segs = []

            for data in batch_data:
                image_data, word_vec, image, sentence, seg_map = get_random_data(data, self.input_shape,
                                                                                 self.word_embed, self.config,
                                                                                 train_mode=False)  # box is [1,5]
                sentences.extend(sentence)
                word_vecs.extend(word_vec)
                # evaluate each sentence corresponding to the same image
                for ___ in range(len(sentence)):
                    images.append(image_data)
                    images_ori.append(image)
                    files_id.append(img_id)
                    gt_segs.append(seg_map)
                    img_id += 1

            images = np.array(images)
            word_vecs = np.array(word_vecs)
            mask_outs = self.model.predict_on_batch([images, word_vecs])
            mask_outs = self.sigmoid_(mask_outs)  # logit to sigmoid
            batch_size = mask_outs.shape[0]
            for i in range(batch_size):
                ih = gt_segs[i].shape[0]
                iw = gt_segs[i].shape[1]
                w, h = self.input_shape
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)
                dx = (w - nw) // 2
                dy = (h - nh) // 2

                pred_seg = mask_outs[i, :, :, 0]

                pred_seg = cv2.resize(pred_seg, self.input_shape)
                pred_seg = pred_seg[dy:nh + dy, dx:nw + dx, ...]
                pred_seg = cv2.resize(pred_seg, (gt_segs[i].shape[1], gt_segs[i].shape[0]))
                pred_seg = np.reshape(pred_seg, [pred_seg.shape[0], pred_seg.shape[1], 1])

                self.update_confusion(confusion, gt_segs[i], pred_seg, self.seg_min_overlap)

                if self.log_images:
                    os.makedirs('log/out_img', exist_ok=True)
                    sent = sentences[i]['sent']
                    cv2.imwrite('log/out_img/'+str(files_id[i])+'_'+sent+'_pred.png', pred_seg * 255)
                    cv2.imwrite('log/out_img/'+str(files_id[i])+'_'+sent+'_gt.png', gt_segs[i])
                    cv2.imwrite('log/out_img/'+str(files_id[i])+'_'+sent+'_img.png', images[i][dy:nh + dy, dx:nw + dx, ...] * 255)

        return self.compute_metrics(confusion)

    def update_confusion(self, confusion, gt, pred, thresh=0.5):
        pred_fg = np.array(pred > thresh)
        gt_fg = gt > 0.

        confusion['tp'] += int(np.logical_and(pred_fg, gt_fg).sum())
        confusion['fp'] += int(np.logical_and(pred_fg, np.logical_not(gt_fg)).sum())
        confusion['fn'] += int(np.logical_and(np.logical_not(pred_fg), gt_fg).sum())
        confusion['tn'] += int(np.logical_and(np.logical_not(pred_fg), np.logical_not(gt_fg)).sum())

    def compute_metrics(self, confusion):
        tp = confusion['tp']
        fp = confusion['fp']
        fn = confusion['fn']
        tn = confusion['tn']
        eps = 1e-10

        fg_iou = tp / (tp + fp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        recall = tp / (tp + fn + eps)
        bg_iou = tn / (tn + fp + fn + eps)
        bg_acc = tn / (tn + fp + eps)
        miou = (fg_iou + bg_iou) / 2.0
        macc = (recall + bg_acc) / 2.0

        return {
            'IoU': float(fg_iou),
            'Dice': float(dice),
            'Recall': float(recall),
            'mIoU': float(miou),
            'mACC': float(macc),
        }

    def sigmoid_(self, x):
        return (1. + 1e-9) / (1. + np.exp(-x) + 1e-9)

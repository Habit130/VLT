import os

import cv2
import keras
import numpy as np
import spacy


class Generator(keras.utils.Sequence):
    """Sequence-based data loader for the local JSON dataset."""

    def __init__(
        self,
        data,
        config,
        shuffle=True,
        train_mode=True,
    ):
        self.shuffle = shuffle
        self.data = data
        self.config = config
        self.train_mode = train_mode
        self.batch_size = config.batch_size
        self.embed = spacy.load(config.word_embed)
        self.input_shape = (config.input_size, config.input_size)
        validate_dataset(self.data, self.config)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)
        self.group()

    def size(self):
        return len(self.data)

    def group(self):
        self.groups = [
            [self.data[x % len(self.data)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(self.data), self.batch_size)
        ]

    def __len__(self):
        return len(self.groups)

    def get_batch(self, datas):
        size = len(datas)
        image_data = np.empty([size, self.input_shape[0], self.input_shape[1], 3], dtype=np.float32)
        word_data = np.empty([size, self.config.word_len, self.config.embed_dim], dtype=np.float32)
        seg_data = np.empty(
            [size, self.input_shape[0] // self.config.seg_out_stride, self.input_shape[1] // self.config.seg_out_stride, 1],
            dtype=np.float32,
        )

        for i, data in enumerate(datas):
            image, word_vec, seg_map = get_random_data(
                data,
                self.input_shape,
                self.embed,
                self.config,
                train_mode=self.train_mode,
            )
            word_data[i] = word_vec
            image_data[i] = image
            seg_data[i] = seg_map

        return image_data, word_data, seg_data

    def __getitem__(self, index):
        group = self.groups[index]
        image_data, word_data, seg_data = self.get_batch(group)
        return [image_data, word_data, seg_data], np.zeros((len(group),), dtype=np.float32)


def dataset_root(config):
    return getattr(config, "dataset_root", ".")


def caption_index(config):
    return int(getattr(config, "caption_index", 2))


def resolve_path(root, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(root, path_value))


def select_caption(sample, config):
    captions = sample.get("caption")
    if not isinstance(captions, list) or not captions:
        raise ValueError("Dataset item '{}' is missing a non-empty 'caption' list.".format(sample.get("id", "<unknown>")))

    index = caption_index(config)
    if index < 0 or index >= len(captions):
        raise ValueError(
            "Dataset item '{}' does not contain caption index {}.".format(sample.get("id", "<unknown>"), index)
        )
    return captions[index]


def validate_dataset(data, config):
    if not data:
        raise ValueError("Dataset is empty.")

    root = dataset_root(config)
    for sample in data:
        if "image" not in sample or "mask" not in sample:
            raise ValueError("Dataset item '{}' is missing 'image' or 'mask'.".format(sample.get("id", "<unknown>")))
        select_caption(sample, config)
        image_path = resolve_path(root, sample["image"])
        mask_path = resolve_path(root, sample["mask"])
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image file not found: {}".format(image_path))
        if not os.path.exists(mask_path):
            raise FileNotFoundError("Mask file not found: {}".format(mask_path))


def qlist_to_vec(max_length, sentence, embed, emb_size=300):
    tokens = sentence.split()
    glove_matrix = np.zeros((max_length, emb_size), dtype=np.float32)
    q_len = min(max_length, len(tokens))

    for i in range(q_len):
        glove_matrix[i, :] = embed(tokens[i]).vector

    return glove_matrix


def load_raw_sample(ref, config):
    root = dataset_root(config)
    sentence = select_caption(ref, config)

    image_path = resolve_path(root, ref["image"])
    mask_path = resolve_path(root, ref["mask"])

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Failed to read image: {}".format(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    seg_map = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    if seg_map is None:
        raise FileNotFoundError("Failed to read mask: {}".format(mask_path))
    seg_map = (seg_map > 0).astype(np.float32)

    return image, seg_map, sentence


def get_random_data(ref, input_shape, embed, config, train_mode=True):
    h, w = input_shape
    image, seg_map, sentence = load_raw_sample(ref, config)
    word_vec = qlist_to_vec(config.word_len, sentence, embed, emb_size=config.embed_dim)

    ih, iw, _ = image.shape
    if not train_mode:
        ori_image = image.copy()

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    image_data = np.full((h, w, 3), 0.5, dtype=np.float32)
    image_data[dy:dy + nh, dx:dx + nw, :] = image / 255.0

    resized_mask = cv2.resize(seg_map, (nw, nh), interpolation=cv2.INTER_NEAREST)
    seg_canvas = np.zeros((h, w), dtype=np.float32)
    seg_canvas[dy:dy + nh, dx:dx + nw] = resized_mask

    if train_mode:
        seg_map_data = cv2.resize(
            seg_canvas,
            (w // config.seg_out_stride, h // config.seg_out_stride),
            interpolation=cv2.INTER_NEAREST,
        )
        seg_map_data = seg_map_data[:, :, None].astype(np.float32)
        return image_data, word_vec, seg_map_data

    return image_data, word_vec, ori_image, sentence, seg_map[:, :, None].astype(np.float32)

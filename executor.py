import os
from abc import abstractmethod
from datetime import datetime

import keras
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from callbacks.eval import Evaluate
from callbacks.learning_scheduler import LearningRateScheduler, lr_step_decay
from loader.loader import Generator
from model.vlt_model import yolo_body, yolo_loss


class Executor(object):
    def __init__(self, config, GPUS=1, debug=False):
        self.config = config
        self.debug = debug
        self.GPUS = GPUS
        self.input_shape = (self.config.input_size, self.config.input_size, 3)
        self.word_len = self.config.word_len
        self.embed_dim = self.config.embed_dim
        self.seg_out_stride = self.config.seg_out_stride
        self.start_epoch = self.config.start_epoch
        self.n_freeze = 185 + 12

        self.dataset = {}
        self.dataset_len = {}
        self.load_data()

        self.yolo_model, self.yolo_body, self.yolo_body_single = self.create_model()
        self.callbacks = self.build_callbacks()

    def create_model(self):
        print("Creating model...")
        K.clear_session()
        image_input = Input(shape=self.input_shape)
        q_input = Input(shape=[self.word_len, self.embed_dim], name="q_input")
        h, w, _ = self.input_shape

        seg_gt = Input(shape=(h // self.seg_out_stride, w // self.seg_out_stride, 1))
        model_body = yolo_body(image_input, q_input, self.config)
        print("Loading model...")
        self.load_model(model_body)

        model_loss = Lambda(
            yolo_loss,
            output_shape=(1,),
            name="yolo_loss",
            arguments={"batch_size": self.config.batch_size},
        )([model_body.output, seg_gt])

        model = Model([model_body.input[0], model_body.input[1], seg_gt], model_loss)
        print("Model created.")
        return model, model_body, model_body

    def load_dataset(self, split):
        import json

        with open(self.config[split], "r", encoding="utf-8") as f:
            data_lines = json.load(f)
        if self.debug:
            data_lines = data_lines[:50]
        set_num = len(data_lines)
        print("Dataset Loaded: %s,  Len: %d" % (split, set_num))
        return data_lines, set_num

    @abstractmethod
    def build_callbacks(self):
        pass

    @abstractmethod
    def load_model(self, model_body):
        pass

    @abstractmethod
    def load_data(self):
        pass


class Trainer(Executor):
    def __init__(self, config, log_path, verbose=False, **kwargs):
        self.log_path = log_path
        self.verbose = verbose
        self.model_path = os.path.join(self.log_path, "models")
        self.last_model_path = os.path.join(self.model_path, "last.weights.h5")
        self.final_model_path = os.path.join(self.model_path, "final.weights.h5")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with open(os.path.join(self.model_path, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(config.dump())

        timestr = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.tb_path = os.path.join(self.log_path, timestr)
        if not os.path.exists(self.tb_path):
            os.makedirs(self.tb_path)
        with open(os.path.join(self.tb_path, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(config.dump())

        super(Trainer, self).__init__(config, **kwargs)

    def load_model(self, model_body):
        resume_path = getattr(self.config, "resume_model", "")
        if resume_path:
            model_body.load_weights(resume_path, by_name=False, skip_mismatch=False)
            print("Resuming full model weights from {}.".format(resume_path))
        else:
            path = self.config.pretrained_weights
            model_body.load_weights(path, by_name=True, skip_mismatch=True)
            print("Loading pretrained weights from {}.".format(path))

        if self.config.free_body in [1, 2]:
            num = (self.n_freeze, len(model_body.layers) - 3)[self.config.free_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print("Freeze the first {} layers of total {} layers.".format(num, len(model_body.layers)))

    def load_data(self):
        self.dataset["train"], self.dataset_len["train"] = self.load_dataset("train_set")
        self.train_generator = Generator(self.dataset["train"], self.config)

    def build_callbacks(self):
        callbacks = []
        logging = TensorBoard(log_dir=self.tb_path)
        callbacks.append(logging)

        checkpoint_last = ModelCheckpoint(
            self.last_model_path,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
        )
        callbacks.append(checkpoint_last)

        lr_schedule = LearningRateScheduler(
            lr_step_decay(self.config.lr, self.config.steps),
            logging,
            verbose=1,
            init_epoch=self.config.start_epoch,
        )
        callbacks.append(lr_schedule)
        return callbacks

    def train(self):
        print("Compiling model...")
        self.yolo_model.compile(
            loss={"yolo_loss": lambda y_true, y_pred: y_pred},
            optimizer=Adam(learning_rate=self.config.lr),
        )

        use_multiprocessing = self.config.workers > 0

        print("Starting training:")
        self.yolo_model.fit(
            self.train_generator,
            callbacks=self.callbacks,
            epochs=self.config.epoches,
            initial_epoch=self.config.start_epoch,
            verbose=True,
            workers=self.config.workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=self.config.max_queue_size,
        )
        self.yolo_body_single.save_weights(self.final_model_path)
        print("Saved final weights to {}.".format(self.final_model_path))


class Tester(Executor):
    def __init__(self, config, **kwargs):
        super(Tester, self).__init__(config, **kwargs)

    def build_callbacks(self):
        return None

    def load_data(self):
        self.dataset["test"], self.dataset_len["test"] = self.load_dataset("evaluate_set")

    def load_model(self, model_body):
        model_body.load_weights(self.config.evaluate_model, by_name=False, skip_mismatch=False)
        print("Load weights {}.".format(self.config.evaluate_model))

    def eval(self):
        evaluator = Evaluate(self.yolo_body, self.dataset["test"], self.config)
        metrics = evaluator.evaluate()
        json_path, txt_path = evaluator.save(metrics, "result", self.config.evaluate_model, self.config.evaluate_set)
        print("Saved test metrics to {} and {}.".format(json_path, txt_path))


class Debugger(Executor):
    def __init__(self, config, **kwargs):
        self.config = config
        self.debug = True
        self.dataset, self.dataset_len = self.load_dataset("train_set")
        self.train_generator = Generator(self.dataset, self.config)
        self.train_generator.__getitem__(0)
        kwargs.update({"GPUS": 1, "debug": True})
        super(Debugger, self).__init__(config, **kwargs)

    def build_callbacks(self):
        return None

    def load_data(self):
        return None

    def load_model(self, model_body):
        return None

    def run(self):
        self.yolo_model.summary()
        self.yolo_model.compile(
            loss={"yolo_loss": lambda y_true, y_pred: y_pred},
            optimizer=Adam(learning_rate=self.config.lr),
        )

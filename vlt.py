import argparse
import os

import numpy as np
import tensorflow as tf
from yacs.config import CfgNode as CN

from executor import Debugger, Tester, Trainer

MODES = ["train", "test", "debug"]

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=MODES)
parser.add_argument("config_path")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

assert args.phase in MODES

if not args.verbose:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

tf.compat.v1.disable_eager_execution()

with open("config/base.yaml", "r", encoding="utf-8") as f:
    _C = CN.load_cfg(f)


if __name__ == "__main__":
    config = _C.clone()
    config.merge_from_file(args.config_path)
    config.freeze()
    print(config)
    print("\n\n--------------------------")
    print("PHASE:{}\n".format(args.phase))

    config_basename = os.path.basename(args.config_path)
    config_name = os.path.splitext(config_basename)[0]
    log_path = config.log_path + "_" + config_name

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    tf.compat.v1.set_random_seed(config.seed)

    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_names = [device.name for device in gpu_devices]
    gpu_count = len(gpu_devices)
    print("{} GPUs detected:".format(gpu_count))
    print(gpu_names)

    if args.phase == "train":
        trainer = Trainer(config, log_path, GPUS=max(gpu_count, 1), debug=args.debug, verbose=args.verbose)
        trainer.train()
    elif args.phase == "test":
        tester = Tester(config, GPUS=max(gpu_count, 1), debug=args.debug)
        tester.eval()
    elif args.phase == "debug":
        debugger = Debugger(config)
        debugger.run()
    print("Exited.")

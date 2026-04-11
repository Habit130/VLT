import argparse
import os
import numpy as np
import tensorflow as tf
from yacs.config import CfgNode as CN

from executor import Tester, Trainer, Debugger
from runtime_utils import ensure_dir

MODES = ['train', 'test', 'debug']

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=MODES)
parser.add_argument('config_path')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

assert (args.phase in MODES)

if not args.verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.compat.v1.disable_eager_execution()

with open('config/base.yaml', 'r') as f:
    _C = CN.load_cfg(f)


if __name__ == "__main__":
    config = _C.clone()
    config.merge_from_file(args.config_path)
    config.freeze()
    print(config)
    print('\n\n--------------------------')
    print('PHASE:{}\n'.format(args.phase))

    config_basename = os.path.basename(args.config_path)
    config_name = os.path.splitext(config_basename)[0]
    log_path = config.log_path + '_' + config_name

    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)

    gpu_devices = [x.name for x in tf.config.list_logical_devices('GPU')]

    GPU_COUNTS = len(gpu_devices)
    print("{} GPUs detected:".format(GPU_COUNTS))
    print(gpu_devices)

    if __name__ == "__main__":
        if (args.phase == 'train'):
            ensure_dir('log')
            trainer = Trainer(config, log_path, GPUS=GPU_COUNTS, debug=args.debug, verbose=args.verbose)
            trainer.train()
        elif (args.phase == 'test'):
            ensure_dir('result')
            tester = Tester(config, GPUS=GPU_COUNTS, debug=args.debug)
            tester.eval()
        elif (args.phase == 'debug'):
            debugger = Debugger(config)
            debugger.run()
        print('Exited.')

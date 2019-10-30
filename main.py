from training.training_loop import training_loop
from utils.training_utils import *
from argparse import ArgumentParser


usage = 'Parser for all sample.'
parser = ArgumentParser(description=usage)
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size for sample')

args = parser.parse_args()
config = Config()
config.make_task_dir()
config.make_task_log()
config.set(gpu_nums=8, batch_size=args.batch_size)
config.write_config_and_gin()
training_loop(config=config)
config.terminate()

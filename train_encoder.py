from training.train_encoder import training_loop
from utils.training_utils import *
from argparse import ArgumentParser


usage = 'Parser for all sample.'
parser = ArgumentParser(description=usage)
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size for sample')
parser.add_argument('--seed', type=int, default=23,
                    help='seed for np')
parser.add_argument('--gpu_nums', type=int, default=1,
                    help='seed for np')
parser.add_argument('--model_dir_root', type=str, default='/gdata/fengrl/SSGAN',
                    help='seed for np')
parser.add_argument('--h5root', type=str, default='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5',
                    help='seed for np')
parser.add_argument('--restore_g_dir', type=str, default='/ghome/fengrl/gen_ckpt/gen-0',
                    help='seed for np')
parser.add_argument('--restore_d_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
                    help='seed for np')
args = vars(parser.parse_args())
config = Config()
config.set(**args)
config.make_task_dir()
config.make_task_log()
config.write_config_and_gin()
training_loop(config=config)
config.terminate()

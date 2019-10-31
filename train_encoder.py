from utils.training_utils import *
from argparse import ArgumentParser
from importlib import import_module


usage = 'Parser for all sample.'
parser = ArgumentParser(description=usage)
parser.add_argument('--model', type=str, default='train_vgg',
                    help='seed for np')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='batch size for sample')
parser.add_argument('--seed', type=int, default=23,
                    help='seed for np')
parser.add_argument('--gpu_nums', type=int, default=4,
                    help='seed for np')
parser.add_argument('--model_dir_root', type=str, default='/gdata1/fengrl/SSGAN',
                    help='seed for np')
parser.add_argument('--h5root', type=str, default='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5',
                    help='seed for np')
parser.add_argument('--finalize', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--restore_g_dir', type=str, default='/ghome/fengrl/gen_ckpt/gen-0',
                    help='seed for np')
parser.add_argument('--restore_d_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
                    help='seed for np')
parser.add_argument('--restore_v_dir', type=str, default='/gdata1/fengrl/SSGAN/00020-invSSGAN/vgg-0',
                    help='seed for np')

args = vars(parser.parse_args())

model = import_module('training.' + args['model'])

config = Config()
config.set(**args)
config.make_task_dir()
config.make_task_log()
config.write_config_and_gin()
model.training_loop(config=config)
config.terminate()

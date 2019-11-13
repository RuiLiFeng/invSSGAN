from utils.training_utils import *
from argparse import ArgumentParser
from importlib import import_module


usage = 'Parser for all sample.'
parser = ArgumentParser(description=usage)
parser.add_argument('--model', type=str, default='ReLearning',
                    help='seed for np')
parser.add_argument('--labeled_per_class', type=int, default=10,
                    help='seed for np')
parser.add_argument('--load_in_mem', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--resume', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--resume_assgin', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--resume_ebn', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--load_num', type=int, default=None,
                    help='seed for np')
parser.add_argument('--disc_iter', type=int, default=2,
                    help='seed for np')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='batch size for sample')
parser.add_argument('--seed', type=int, default=23,
                    help='seed for np')
parser.add_argument('--gpu_nums', type=int, default=4,
                    help='seed for np')
parser.add_argument('--eval_per_steps', type=int, default=2000,
                    help='seed for np')
parser.add_argument('--save_per_steps', type=int, default=2000,
                    help='seed for np')
parser.add_argument('--lr_decay_step', type=int, default=60000,
                    help='seed for np')
parser.add_argument('--r_loss_scale', type=float, default=0.4,
                    help='seed for np')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='seed for np')
parser.add_argument('--g_loss_scale', type=float, default=0.03,
                    help='seed for np')
parser.add_argument('--s_loss_scale', type=float, default=0.01,
                    help='seed for np')
parser.add_argument('--alpha', type=float, default=0.02,
                    help='seed for np')
parser.add_argument('--beta', type=float, default=0.1,
                    help='seed for np')
parser.add_argument('--model_dir_root', type=str, default='/gdata1/fengrl/ReLearning',
                    help='seed for np')
parser.add_argument('--h5root', type=str, default='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5',
                    help='seed for np')
parser.add_argument('--eval_h5root', type=str, default='/gpub/temp/imagenet2012/hdf5/ILSVRC128_eval.hdf5',
                    help='seed for np')
parser.add_argument('--finalize', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--restore_g_dir', type=str, default='/ghome/fengrl/gen_ckpt/gen-0',
                    help='seed for np')
parser.add_argument('--restore_d_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
                    help='seed for np')
parser.add_argument('--restore_assgin_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
                    help='seed for np')
parser.add_argument('--restore_v_dir', type=str, default='/gdata1/fengrl/SSGAN/00036-invSSGAN/vgg.ckpt-12000',
                    help='seed for np')
parser.add_argument('--restore_ebn_dir', type=str, default='/gdata1/fengrl/SSGAN/00036-invSSGAN/vgg.ckpt-12000',
                    help='seed for np')
parser.add_argument('--task_name', type=str, default='ReLearn',
                    help='seed for np')
parser.add_argument('--triple_margin', type=float, default=0.1,
                    help='seed for np')


args = vars(parser.parse_args())

model = import_module('ReLearning.' + args['model'])

tf.config.optimizer.set_jit(True)

config = Config()
config.set(**args)
config.make_task_dir()
config.make_task_log()
config.write_config_and_gin()
model.training_loop(config=config)
config.terminate()

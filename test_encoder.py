from utils.training_utils import *
from argparse import ArgumentParser
from importlib import import_module
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan, arch_ops
import logging
from network.resnet import ImagenetModel
import h5py as h5


usage = 'Parser for all sample.'
parser = ArgumentParser(description=usage)
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for sample')
parser.add_argument('--seed', type=int, default=23,
                    help='seed for np')
parser.add_argument('--save_output_dir', type=str, default='/gdata/fengrl/SSGAN/test',
                    help='seed for np')
parser.add_argument('--h5root', type=str, default='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5',
                    help='seed for np')
parser.add_argument('--finalize', action='store_true', default=False,
                    help='seed for np')
parser.add_argument('--restore_g_dir', type=str, default='/ghome/fengrl/gen_ckpt/gen-0',
                    help='seed for np')
parser.add_argument('--restore_d_dir', type=str, default='/ghome/fengrl/disc_ckpt/disc-0',
                    help='seed for np')
parser.add_argument('--restore_e_dir', type=str, default='/gdata1/fengrl/SSGAN/00042-train_vgg/vgg.ckpt-48000',
                    help='seed for np')
parser.add_argument('--encoder_name', type=str, default='vae_alter',
                    help='seed for np')
parser.add_argument('--save_name', type=str, default='vae-16000',
                    help='seed for np')
parser.add_argument('--embed_dir', type=str, default='/ghome/fengrl/ssgan/invSSGAN/G_embed.npy',
                    help='seed for np')


args = parser.parse_args()

Encoder = ImagenetModel(resnet_size=50, num_classes=120, name=args.encoder_name)
Generator = resnet_biggan.Generator(image_shape=[128, 128, 3], embed_y=False,
                                    embed_z=False,
                                    batch_norm_fn=arch_ops.self_modulated_batch_norm,
                                    spectral_norm=True)
G_embed_np = np.load(args.embed_dir)
G_embed = tf.convert_to_tensor(G_embed_np, dtype=tf.float32, name='G_embed')

z = tf.random.normal([args.batch_size, 120], stddev=1.0, name='z')
wz = tf.matmul(z, G_embed)
index = np.arange(1000000)
np.random.shuffle(index)
index = index[:args.batch_size]
index.sort()
with h5.File(args.h5root, 'r') as f:
    img = f['imgs'][list(index)].transpose([0, 2, 3, 1]) / 255.0
x = tf.placeholder(tf.float32, [args.batch_size, 128, 128, 3])
w = Encoder(x, training=False)
fake = Generator(w, y=None, is_training=True)
fake_ss = Generator(wz, y=None, is_training=True)

init = [tf.global_variables_initializer()]
restore_g = [v for v in tf.global_variables() if 'opt' not in v.name
             and 'beta1_power' not in v.name
             and 'beta2_power' not in v.name
             and 'generator' in v.name]
saver_g = tf.train.Saver(restore_g, restore_sequentially=True)
saver_e = tf.train.Saver(Encoder.trainable_variables, restore_sequentially=True)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)
    print("Restore generator...")
    saver_g.restore(sess, args.restore_g_dir)
    saver_e.restore(sess, args.restore_e_dir)
    fake_img = sess.run(fake, {x: img})
    fake_ss_img = sess.run(fake_ss)
    save_image_grid(img, args.save_output_dir + '/' + args.save_name + 'reals_imagenet.png')
    save_image_grid(fake_img, args.save_output_dir + '/' + args.save_name + 'fakes_imagenet.png')
    save_image_grid(fake_ss_img, args.save_output_dir + '/' + args.save_name + 'fakes_ss.png')
    print('Done! See outputs in %s' % (args.save_output_dir + '/' + args.save_name))

from datasets import get_dataset
from utils.training_utils import *
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan, arch_ops
import logging
from tensorflow import keras
import tensorflow_datasets as tfds
import h5py as h5
from tqdm import tqdm
import numpy as np


def load_from_h5(root='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5', batch_size=256, shuffle_batch_size=10000):
    with h5.File(root, 'r') as f:
        img = (f['imgs'][1:] / 255.0).astype(np.float32).transpose([0,2,3,1])
        label = f['labels'][1:]
    dset = tf.data.Dataset.from_tensor_slices((img, label))
    dset = dset.shuffle(shuffle_batch_size).batch(batch_size).prefetch(tf.contrib.data.AUTOTUNE)
    return dset


class np_dataset(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5', batch_size=256):
        print('Loading data root %s into memory...' % root)
        with h5.File(root, 'r') as f:
            self.img = f['imgs'][1:]
            self.label = f['labels'][1:]
        self.num_imgs = len(self.label)
        self.index = np.arange(self.num_imgs)
        self.batch_size = batch_size

    def gen(self):
        for i in self.index:
            yield self.img[i]

    def __len__(self):
        return self.num_imgs


def parser_fn(img):
    return img.transpose([1, 2, 0]).astype(np.float32) / 255.0


def build_np_dataset(root, batch_size, gpu_nums):
    h5_dset = np_dataset(root, batch_size)
    dset = tf.data.Dataset.from_generator(h5_dset.gen(), tf.float32)
    dset = dset.map(map_func=parser_fn, num_parallel_calls=3 * gpu_nums).shuffle(len(h5_dset)).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset


def compute_loss(train_step, strategy):
    e_loss = strategy.experimental_run_v2(train_step, ())
    mean_e_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, e_loss, axis=None)
    return mean_e_losses


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()
    print('Loading Imagenet2012 dataset...')
    # dataset = load_from_h5(root=config.h5root, batch_size=config.batch_size)
    dataset = build_np_dataset(root=config.h5root, batch_size=config.batch_size, gpu_nums=config.gpu_nums)
    dataset = dataset.make_initializable_iterator()
    with strategy.scope():
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        # dataset = get_dataset(name=config.dataset,
        #                       seed=config.seed).train_input_fn(params=data_iter_params)
        # dataset = strategy.experimental_distribute_dataset(dataset)
        # data_iter = dataset.make_initializable_iterator()
        print("Constructing networks...")
        # img = tf.placeholder(tf.float32, [None, 128, 128, 3])
        fixed_x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        img = dataset.get_next()
        Encoder = keras.applications.ResNet50(weights=None, classes=config.dim_z)
        VGG_alter = keras.applications.ResNet50(weights=None, classes=config.dim_z)
        Generator = resnet_biggan.Generator(image_shape=[128, 128, 3], embed_y=False,
                                            embed_z=False,
                                            batch_norm_fn=arch_ops.self_modulated_batch_norm,
                                            spectral_norm=True)
        Discriminator = resnet_biggan.Discriminator(spectral_norm=True, project_y=False)
        learning_rate = tf.train.exponential_decay(0.0001, global_step, 150000 / config.gpu_nums,
                                                   0.8, staircase=False)
        E_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='e_opt', beta1=config.beta1)
        # D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate * 5, name='d_opt', beta1=config.beta1)

        print("Building tensorflow graph...")
        w = Encoder(img)
        x = Generator(w, y=None, is_training=True)
        # _, real_logits, _ = Discriminator(img, y=None, is_training=True)
        _, fake_logits, _ = Discriminator(x, y=None, is_training=True)
        # real_logits = fp32(real_logits)
        fake_logits = fp32(fake_logits)
        with tf.variable_scope('recon_loss'):
            recon_loss_pixel = tf.reduce_mean(tf.square(x - img))
            adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_logits)) * config.g_loss_scale
            vgg_real = VGG_alter(img)
            vgg_fake = VGG_alter(x)
            feature_scale = tf.cast(tf.reduce_prod(vgg_real.shape[1:]), dtype=tf.float32)
            vgg_loss = config.r_loss_scale * tf.nn.l2_loss(vgg_fake - vgg_real) / (config.batch_size * feature_scale)
            e_loss = recon_loss_pixel + adv_loss + vgg_loss
        # with tf.variable_scope('d_loss'):
        #     d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_logits))
        #     d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_logits))
        #     d_loss = d_loss_real + d_loss_fake

        add_global = global_step.assign_add(1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        def train_step():
            with tf.control_dependencies([add_global] + update_ops):
                E_opt = E_solver.minimize(e_loss, var_list=Encoder.trainable_variables)
                with tf.control_dependencies([E_opt]):
                    return tf.identity(e_loss)
        e_loss = compute_loss(train_step, strategy)
        print("Building eval module...")
        with tf.init_scope():
            fixed_w = Encoder(fixed_x)
            fixed_sample = Generator(z=fixed_w, y=None, is_training=False)
        print('Building init module...')
        with tf.init_scope():
            init = [tf.global_variables_initializer(), dataset.initializer]
            restore_g = [v for v in tf.global_variables() if 'opt' not in v.name
                         and 'beta1_power' not in v.name
                         and 'beta2_power' not in v.name
                         and 'generator' in v.name]
            restore_d = [v for v in tf.global_variables() if 'opt' not in v.name
                         and 'beta1_power' not in v.name
                         and 'beta2_power' not in v.name
                         and 'discriminator' in v.name]
            saver_g = tf.train.Saver(restore_g, restore_sequentially=True)
            saver_d = tf.train.Saver(restore_d, restore_sequentially=True)
            saver_v = tf.train.Saver(VGG_alter.trainable_variables, restore_sequentially=True)
            saver_e = tf.train.Saver(Encoder.trainable_variables, restore_sequentially=True)
        print("Start training...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            print("Restore generator and discriminator...")
            saver_g.restore(sess, config.restore_g_dir)
            saver_d.restore(sess, config.restore_d_dir)
            saver_v.restore(sess, config.restore_v_dir)
            timer.update()
            fixed_img = sess.run(dataset.get_next())
            save_image_grid(fixed_img, filename=config.model_dir + '/reals.png')
            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)

            print("Start iterations...")
            for iteration in range(config.total_step):
                e_loss_, adv_loss_, recon_loss_pixel_, vgg_loss_, lr_ = sess.run(
                    [e_loss, adv_loss, recon_loss_pixel, vgg_loss, learning_rate])
                if iteration % config.print_loss_per_steps == 0:
                    timer.update()
                    print("step %d, e_loss %f, adv_loss %f, recon_loss_pixel %f, vgg_loss %f, "
                          "learning_rate % f, consuming time %s" %
                          (iteration, e_loss_, adv_loss_, recon_loss_pixel_, vgg_loss_,
                           lr_, timer.runing_time_format))
                if iteration % config.eval_per_steps == 0:
                    timer.update()
                    fixed_ = sess.run(fixed_sample, {fixed_x: fixed_img})
                    save_image_grid(fixed_, filename=config.model_dir + '/fakes%06d.png' % iteration)
                if iteration % config.save_per_steps == 0:
                    saver_e.save(sess, save_path=config.model_dir + '/en.ckpt',
                                 global_step=iteration, write_meta_graph=False)


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

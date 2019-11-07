from datasets import get_dataset
from utils.training_utils import *
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan, arch_ops
import logging
from network.resnet import ImagenetModel
import tensorflow_datasets as tfds
import h5py as h5
from tqdm import tqdm
from utils import np_dataset as npdt
import numpy as np


class np_dataset(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5', batch_size=256, load_in_mem=True):
        print('Loading data root %s into memory...' % root)
        self.root = root
        self.load_in_mem = load_in_mem
        self.num_imgs = len(h5.File(root, 'r')['labels'])
        if self.load_in_mem:
            with h5.File(root, 'r') as f:
                self.img = f['imgs'][1:]
                self.label = f['labels'][1:]
        self.index = np.arange(self.num_imgs)
        np.random.shuffle(self.index)
        self.batch_size = batch_size // 4

    def gen_from_mem(self):
        for i in self.index:
            yield self.img[i]

    def gen_from_file(self):
        for i in self.index:
            with h5.File(self.root, 'r') as f:
                yield f['imgs'][i + 1]

    def fixed_sample(self):
        index = np.random.randint(0, self.num_imgs, self.batch_size)
        index.sort()
        if self.load_in_mem:
            return self.img[list(index)].transpose([0, 2, 3, 1]) / 255.0
        else:
            with h5.File(self.root, 'r') as f:
                return f['imgs'][list(index + 1)].transpose([0, 2, 3, 1]) / 255.0

    def __len__(self):
        return self.num_imgs


def parser_fn(img):
    return tf.cast(tf.transpose(img, [1, 2, 0]), tf.float32) / 255.0


def build_np_dataset(root, batch_size, gpu_nums, load_in_mem=True):
    h5_dset = np_dataset(root, batch_size, load_in_mem)
    gen = h5_dset.gen_from_mem if h5_dset.load_in_mem else h5_dset.gen_from_file
    fixed_img = h5_dset.fixed_sample()
    dset = tf.data.Dataset.from_generator(gen, tf.float32, output_shapes=[3, 128, 128])
    print('Making tensorflow dataset with length %d' % len(h5_dset))
    dset = dset.map(map_func=parser_fn, num_parallel_calls=3 * gpu_nums).shuffle(10000).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset, fixed_img


def compute_loss(train_step, data, strategy, training_G):
    loss1, loss2 = strategy.experimental_run_v2(train_step, (data, training_G))
    mean_losses1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss1, axis=None)
    mean_losses2 = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss2, axis=None)
    return mean_losses1, mean_losses2


class dense(object):
    def __init__(self, out_ch, bias=True, dtype=tf.float32, name='embed', scope='Generator'):
        self.out_ch = out_ch
        self.bias = bias
        self.dtype = dtype
        self.name = name
        self.scope = scope

    def apply(self, x):
        x = arch_ops.linear(x, self.out_ch, scope=self.name, use_sn=False,
                            use_bias=self.bias)
        return x

    def __call__(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.scope, values=[x], reuse=reuse):
            x = self.apply(x)
        return x

    @property
    def trainable_variables(self):
        return [var for var in tf.trainable_variables() if self.scope in var.name]


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()
    print('Loading Imagenet2012 dataset...')
    # dataset = load_from_h5(root=config.h5root, batch_size=config.batch_size)
    dataset, fixed_img = build_np_dataset(root=config.h5root, batch_size=config.batch_size, gpu_nums=config.gpu_nums,
                                          load_in_mem=config.load_in_mem)
    dataset = strategy.experimental_distribute_dataset(dataset)
    dataset = dataset.make_initializable_iterator()
    with strategy.scope():
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        print("Constructing networks...")
        fixed_x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        Generator = resnet_biggan.Generator(image_shape=[128, 128, 3], embed_y=False,
                                            embed_z=False,
                                            batch_norm_fn=arch_ops.self_modulated_batch_norm,
                                            spectral_norm=True)
        Discriminator = resnet_biggan.Discriminator(spectral_norm=True, project_y=False)
        # Despite Z_embed is out of Generator, it is viewed as part of Generator
        Z_embed = dense(120, False, name='embed_z', scope=Generator.name)
        D_embed = dense(120, True, name='embed_d', scope='Embed_D')
        learning_rate = tf.train.exponential_decay(config.lr, global_step, 60000,
                                                   0.8, staircase=False)
        # learning_rate = tf.convert_to_tensor(config.lr, dtype=tf.float32)
        G_solver = tf.train.AdamOptimizer(learning_rate=config.lr / 2, name='g_opt', beta1=0.0, beta2=config.beta2)
        D_solver = tf.train.AdamOptimizer(learning_rate=config.lr * 2, name='d_opt', beta1=0.0, beta2=config.beta2)
        Embed_solver = tf.train.AdamOptimizer(learning_rate=learning_rate * 5, name='d_opt', beta1=0.0, beta2=config.beta2)
        print("Building tensorflow graph...")

        def train_step(image, training_G):
            z = tf.random.normal([config.batch_size // config.gpu_nums, config.dim_z],
                                 stddev=1.0, name='sample_z')
            w = Z_embed(z)
            fake = Generator(w, y=None, is_training=True)
            fake_out, fake_logits, fake_h = Discriminator(x=fake, y=None, is_training=True)
            real_out, real_logits, real_h = Discriminator(x=image, y=None, is_training=True)
            fake_w = D_embed(fake_h)
            real_w = D_embed(real_h)
            # x is the reconstruction of image
            x = Generator(real_w, None, True)
            d_loss, _, _, g_loss = loss_lib.get_losses(d_real=real_out, d_fake=fake_out, d_real_logits=real_logits,
                                                       d_fake_logits=fake_logits)
            with tf.variable_scope('recon_loss'):
                recon_loss_pixel = tf.reduce_mean(tf.square(x - image))
                sample_loss = tf.reduce_mean(tf.square(w - fake_w)) * config.s_loss_scale
            g_final_loss = g_loss + sample_loss * config.alpha
            d_final_loss = d_loss + recon_loss_pixel * config.beta

            if training_G:
                add_global = global_step.assign_add(1)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies([add_global] + update_ops):
                    G_opt = G_solver.minimize(g_final_loss, var_list=Generator.trainable_variables)
                    with tf.control_dependencies([G_opt]):
                        return tf.identity(g_final_loss), tf.identity(sample_loss)
            else:
                add_global = global_step.assign_add(1)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies([add_global] + update_ops):
                    D_opt = D_solver.minimize(d_final_loss, var_list=Generator.trainable_variables)
                    Embed_opt = Embed_solver.minimize(d_final_loss, var_list=D_embed.trainable_variables)
                    with tf.control_dependencies([D_opt, Embed_opt]):
                        return tf.identity(d_final_loss), tf.identity(recon_loss_pixel)
        g_final_loss, s_loss = compute_loss(train_step, dataset.get_next(), strategy, True)
        d_final_loss, r_loss = compute_loss(train_step, dataset.get_next(), strategy, False)
        print("Building eval module...")
        with tf.init_scope():
            _, _, fixed_h = Discriminator(fixed_x, None, True)
            fixed_w = D_embed(fixed_h)
            fixed_sample = Generator(z=fixed_w, y=None, is_training=True)

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
        print("Start training...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            print("Restore generator...")
            saver_g.restore(sess, config.restore_g_dir)
            saver_d.restore(sess, config.restore_d_dir)
            save_image_grid(fixed_img, filename=config.model_dir + '/reals.png')
            timer.update()

            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)

            print("Start iterations...")
            for iteration in range(config.total_step):
                for D_repeat in range(config.disc_iter):
                    d_loss_, r_loss_ = sess.run([d_final_loss, r_loss])
                g_loss_, s_loss_, lr_ = sess.run([g_final_loss, s_loss, learning_rate])
                if iteration % config.print_loss_per_steps == 0:
                    timer.update()
                    print("step %d, g_loss %f, d_loss %f, r_loss %f, s_loss %f, "
                          "learning_rate % f, consuming time %s" %
                          (iteration, g_loss_, d_loss_, r_loss_, s_loss_,
                           lr_, timer.runing_time_format))
                if iteration % config.eval_per_steps == 0:
                    timer.update()
                    fixed_ = sess.run(fixed_sample, {fixed_x: fixed_img})
                    save_image_grid(fixed_, filename=config.model_dir + '/fakes%06d.png' % iteration)
                if iteration % config.save_per_steps == 0:
                    saver_g.save(sess, save_path=config.model_dir + '/gen.ckpt',
                                 global_step=iteration, write_meta_graph=False)
                    saver_d.save(sess, save_path=config.model_dir + '/disc.ckpt',
                                 global_step=iteration, write_meta_graph=False)


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

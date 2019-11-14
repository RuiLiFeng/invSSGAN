from datasets import get_dataset
from utils.training_utils import *
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan_ssgan, arch_ops
import logging
from network.resnet import ImagenetModel
import tensorflow_datasets as tfds
import h5py as h5
from tqdm import tqdm
from utils import np_dataset as npdt
import numpy as np


EPS = 1e-15


class np_dataset(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5', batch_size=256, load_in_mem=True):
        print('Loading data root %s into memory...' % root)
        self.root = root
        self.load_in_mem = load_in_mem
        self.num_imgs = len(h5.File(root, 'r')['labels']) - 1
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
    dset = dset.map(map_func=parser_fn, num_parallel_calls=2 * gpu_nums).shuffle(10000).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset, fixed_img


def compute_loss(train_step, data, strategy):
    e_loss, r_loss, s_loss = strategy.experimental_run_v2(train_step, (data,))
    mean_e_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, e_loss, axis=None)
    mean_r_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, r_loss, axis=None)
    mean_s_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, s_loss, axis=None)
    return mean_e_losses, mean_r_losses, mean_s_losses


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
        Encoder = ImagenetModel(resnet_size=50, num_classes=None, name='vgg_alter')
        Assgin_net = assgin_net(x0_ch=512+256, scope='Assgin')
        BN_net = BNlayer(scope='Ebn', z0_ch=1536*16)
        Generator = resnet_biggan_ssgan.Generator(image_shape=[128, 128, 3], embed_y=False,
                                                  embed_z=False,
                                                  batch_norm_fn=arch_ops.self_modulated_batch_norm,
                                                  spectral_norm=True)
        learning_rate = tf.train.exponential_decay(config.lr, global_step, config.lr_decay_step,
                                                   0.8, staircase=False)
        E_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='e_opt', beta2=config.beta2)
        G_embed_np = np.load('/ghome/fengrl/ssgan/invSSGAN/G_embed.npy')
        G_embed = tf.convert_to_tensor(G_embed_np, dtype=tf.float32, name='G_embed')
        print("Building tensorflow graph...")

        def train_step(image):
            sample_z = tf.random.normal([config.batch_size // config.gpu_nums, config.dim_z],
                                        stddev=1.0, name='sample_z')
            sample_w = tf.matmul(sample_z, G_embed, name='sample_w')
            sample_img, sample_w_out = Generator(sample_w, y=None, is_training=True)
            ww_ = Encoder(sample_img, training=True)
            ww_ = Assgin_net(ww_)
            ww_ = BN_net(ww_, is_training=True)

            w = Encoder(image, training=True)
            w = Assgin_net(w)
            w = BN_net(w, is_training=True)
            x, _ = Generator(w, y=None, is_training=True)
            with tf.variable_scope('recon_loss'):
                recon_loss_pixel = tf.reduce_mean(tf.square(x - image))
                sample_loss = tf.reduce_mean(tf.square(ww_[:, :1536*16] - sample_w_out[:, :1536*16])) * 0.7
                sample_loss += tf.reduce_mean(tf.square(ww_[:, 1536*16:] - sample_w_out[:, 1536*16:])) * 0.3
                e_loss = recon_loss_pixel + sample_loss * config.s_loss_scale

            add_global = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([add_global] + update_ops):
                E_opt = E_solver.minimize(e_loss,
                                          var_list=Encoder.trainable_variables + Assgin_net.trainable_variables + BN_net.trainable_variables)
                with tf.control_dependencies([E_opt]):
                    return tf.identity(e_loss), tf.identity(recon_loss_pixel), tf.identity(sample_loss)
        e_loss, r_loss, s_loss = compute_loss(train_step, dataset.get_next(), strategy)
        print("Building eval module...")
        with tf.init_scope():
            fixed_w = Encoder(fixed_x, training=True)
            fixed_w = Assgin_net(fixed_w)
            fixed_w = BN_net(fixed_w, is_training=True)
            fixed_sample, _ = Generator(z=fixed_w, y=None, is_training=True)

        print('Building init module...')
        with tf.init_scope():
            init = [tf.global_variables_initializer(), dataset.initializer]
            restore_g = [v for v in tf.global_variables() if 'opt' not in v.name
                         and 'beta1_power' not in v.name
                         and 'beta2_power' not in v.name
                         and 'generator' in v.name]
            saver_g = tf.train.Saver(restore_g, restore_sequentially=True)
            saver_e = tf.train.Saver(Encoder.trainable_variables, restore_sequentially=True)
            saver_assgin = tf.train.Saver(Assgin_net.trainable_variables, restore_sequentially=True)
            saver_bn = tf.train.Saver(BN_net.trainable_variables, restore_sequentially=True)
        print("Start training...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            print("Restore generator...")
            saver_g.restore(sess, config.restore_g_dir)
            if config.resume:
                saver_e.restore(sess, config.restore_v_dir)
                if config.resume_assgin:
                    saver_assgin.restore(sess, config.restore_assgin_dir)
                    if config.resume_ebn:
                        saver_assgin.restore(sess, config.restore_ebn_dir)
            save_image_grid(fixed_img, filename=config.model_dir + '/reals.png')
            timer.update()

            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)

            print("Start iterations...")
            for iteration in range(config.total_step):
                e_loss_, r_loss_, s_loss_, lr_ = sess.run(
                    [e_loss, r_loss, s_loss, learning_rate])
                if iteration % config.print_loss_per_steps == 0:
                    timer.update()
                    print("step %d, e_loss %f, r_loss %f, s_loss %f, "
                          "learning_rate % f, consuming time %s" %
                          (iteration, e_loss_, r_loss_, s_loss_,
                           lr_, timer.runing_time_format))
                if iteration % config.eval_per_steps == 0:
                    timer.update()
                    fixed_ = sess.run(fixed_sample, {fixed_x: fixed_img})
                    save_image_grid(fixed_, filename=config.model_dir + '/fakes%06d.png' % iteration)
                if iteration % config.save_per_steps == 0:
                    saver_e.save(sess, save_path=config.model_dir + '/en.ckpt',
                                 global_step=iteration, write_meta_graph=False)
                    saver_assgin.save(sess, save_path=config.model_dir + '/assgin.ckpt',
                                      global_step=iteration, write_meta_graph=False)
                    saver_bn.save(sess, save_path=config.model_dir + '/bn.ckpt',
                                  global_step=iteration, write_meta_graph=False)


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


class assgin_net(object):
    def __init__(self, x0_ch, name='embed', scope='Encoder'):
        self.x0_ch = x0_ch
        self.x_per_block_ch = (2048 - self.x0_ch) / 5
        self.name = name
        self.scope = scope

    def apply(self, x):
        x0 = x[:, :self.x0_ch]
        x_per_block = tf.split(x[:, self.x0_ch:], 5, axis=1)
        x0 = arch_ops.linear(x0, 128, scope='x0_embed_0', use_sn=True)
        x0 = arch_ops.linear(x0, 16 * 1536, scope='x0_embed_1', use_sn=True)
        for block_idx in range(5):
            x_per_block[block_idx] = arch_ops.linear(x_per_block[block_idx], 20, scope='x%d_embed' % (block_idx + 1),
                                                     use_sn=True)
        x = tf.concat([x0] + x_per_block, axis=1)
        return x

    def __call__(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.scope, values=[x], reuse=reuse):
            x = self.apply(x)
        return x

    @property
    def trainable_variables(self):
        return [var for var in tf.trainable_variables() if self.scope in var.name]


class BNlayer(object):
    def __init__(self, z0_ch, center=True, scale=True, name='batch_norm', scope='Ebn'):
        self.center = center
        self.scale = scale
        self.name = name
        self.scope = scope
        self.z0_ch = z0_ch

    def __call__(self, x, is_training, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.scope, values=[x], reuse=reuse):
            x = self.apply(x, is_training=True)
        return x

    def apply(self, x, is_training):
        x0 = x[:, :self.z0_ch]
        x1 = x[:, self.z0_ch:]
        x1 = arch_ops.batch_norm(x1, is_training, self.center, self.scale, self.name)
        x = tf.concat([x0, x1], axis=1)
        return x

    @property
    def trainable_variables(self):
        return [var for var in tf.trainable_variables() if self.scope in var.name]

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


EPS = 1e-15


class np_dataset(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5', batch_size=256):
        print('Loading data root %s into memory...' % root)
        self.root = root
        with h5.File(root, 'r') as f:
            self.img = f['imgs'][1:]
            self.label = f['labels'][1:]
        self.num_imgs = len(self.label)
        self.index = np.arange(self.num_imgs)
        np.random.shuffle(self.index)
        self.batch_size = batch_size // 4

    def gen(self):
        for i in self.index:
            yield self.img[i]

    def fixed_sample(self):
        index = np.random.randint(0, self.num_imgs, self.batch_size)
        index.sort()
        return self.img[list(index)].transpose([0, 2, 3, 1]) / 255.0

    def __len__(self):
        return self.num_imgs


def parser_fn(img):
    return tf.cast(tf.transpose(img, [1, 2, 0]), tf.float32) / 255.0


def build_np_dataset(root, batch_size, gpu_nums):
    h5_dset = np_dataset(root, batch_size)
    fixed_img = h5_dset.fixed_sample()
    dset = tf.data.Dataset.from_generator(h5_dset.gen, tf.float32, output_shapes=[3, 128, 128])
    print('Making tensorflow dataset with length %d' % len(h5_dset))
    dset = dset.map(map_func=parser_fn, num_parallel_calls=3 * gpu_nums).shuffle(10000).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset, fixed_img


def compute_loss(train_step, data, strategy):
    e_loss, r_loss, s_loss, t_loss = strategy.experimental_run_v2(train_step, (data,))
    mean_e_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, e_loss, axis=None)
    mean_r_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, r_loss, axis=None)
    mean_s_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, s_loss, axis=None)
    mean_t_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, t_loss, axis=None)
    return mean_e_losses, mean_r_losses, mean_s_losses,mean_t_losses


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()
    print('Loading Imagenet2012 dataset...')
    # dataset = load_from_h5(root=config.h5root, batch_size=config.batch_size)
    dataset, fixed_img = build_np_dataset(root=config.h5root, batch_size=config.batch_size, gpu_nums=config.gpu_nums)
    dataset = strategy.experimental_distribute_dataset(dataset)
    dataset = dataset.make_initializable_iterator()
    with strategy.scope():
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        print("Constructing networks...")
        fixed_x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        Encoder = ImagenetModel(resnet_size=50, num_classes=120, name='vgg_alter')
        Generator = resnet_biggan.Generator(image_shape=[128, 128, 3], embed_y=False,
                                            embed_z=False,
                                            batch_norm_fn=arch_ops.self_modulated_batch_norm,
                                            spectral_norm=True)
        learning_rate = tf.train.exponential_decay(config.lr, global_step, 60000,
                                                   0.8, staircase=False)
        E_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='e_opt', beta2=config.beta2)
        G_embed_np = np.load('/ghome/fengrl/ssgan/invSSGAN/G_embed.npy')
        G_embed = tf.convert_to_tensor(G_embed_np, dtype=tf.float32, name='G_embed')
        print("Building tensorflow graph...")

        def train_step(image):
            sample_z = tf.random.normal([config.batch_size // config.gpu_nums, config.dim_z],
                                        stddev=1.0, name='sample_z')
            sample_w = tf.matmul(sample_z, G_embed, name='sample_w')
            sample_img = Generator(sample_w, y=None, is_training=True)
            w = Encoder(image, training=True)
            x = Generator(w, y=None, is_training=True)
            w_ = Encoder(x, training=True)
            ww_ = Encoder(sample_img, training=True)
            with tf.variable_scope('recon_loss'):
                recon_loss_pixel = tf.reduce_mean(tf.square(w_ - w)) / (tf.reduce_mean(tf.square(w)) + EPS)
                trip_loss = triplet_loss(w_, w, config.triple_margin)
                sample_loss = tf.reduce_mean(tf.square(ww_ - sample_w)) / (tf.reduce_mean(tf.square(sample_w)) + EPS)
                e_loss = recon_loss_pixel + sample_loss * config.s_loss_scale + trip_loss

            add_global = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([add_global] + update_ops):
                E_opt = E_solver.minimize(e_loss, var_list=Encoder.trainable_variables)
                with tf.control_dependencies([E_opt]):
                    return tf.identity(e_loss), tf.identity(recon_loss_pixel),\
                           tf.identity(sample_loss), tf.identity(trip_loss)
        e_loss, r_loss, s_loss, t_loss = compute_loss(train_step, dataset.get_next(), strategy)
        print("Building eval module...")
        with tf.init_scope():
            # def eval_fn():
            fixed_w = Encoder(fixed_x, training=True)
            fixed_sample = Generator(z=fixed_w, y=None, is_training=True)
                # return fixed_sample
            # fixed_sample = strategy.experimental_run_v2(eval_fn, ())

        print('Building init module...')
        with tf.init_scope():
            init = [tf.global_variables_initializer(), dataset.initializer]
            restore_g = [v for v in tf.global_variables() if 'opt' not in v.name
                         and 'beta1_power' not in v.name
                         and 'beta2_power' not in v.name
                         and 'generator' in v.name]
            saver_g = tf.train.Saver(restore_g, restore_sequentially=True)
            saver_e = tf.train.Saver(Encoder.trainable_variables, restore_sequentially=True)
        print("Start training...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            print("Restore generator...")
            saver_g.restore(sess, config.restore_g_dir)
            if config.resume:
                saver_e.restore(sess, config.restore_v_dir)
            save_image_grid(fixed_img, filename=config.model_dir + '/reals.png')
            timer.update()

            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)

            print("Start iterations...")
            for iteration in range(config.total_step):
                e_loss_, r_loss_, s_loss_, t_loss_, lr_ = sess.run(
                    [e_loss, r_loss, s_loss, t_loss, learning_rate])
                if iteration % config.print_loss_per_steps == 0:
                    timer.update()
                    print("step %d, e_loss %f, r_loss %f, s_loss %f, t_loss %f "
                          "learning_rate % f, consuming time %s" %
                          (iteration, e_loss_, r_loss_, s_loss_, t_loss_,
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


def triplet_loss(w_, w, margin):
    w_1 = tf.tile(tf.expand_dims(w_, 1), [1, w_.shape[0].value, 1])
    w1 = tf.tile(tf.expand_dims(w, 0), [w.shape[0].value, 1, 1])

    pairwise_dis = tf.reduce_mean(tf.square(w_1 - w1), axis=2) / (tf.reduce_mean(tf.square(w1), axis=2) + EPS)
    p_dis = tf.diag_part(pairwise_dis)
    p_dis = tf.tile(tf.expand_dims(p_dis, 1), [1, p_dis.shape[0].value])
    loss = tf.reduce_mean(tf.nn.relu(p_dis - pairwise_dis + margin * (
            tf.ones_like(p_dis) - tf.eye(p_dis.shape[0].value, p_dis.shape[1].value, dtype=p_dis.dtype))))
    return loss

from datasets import get_dataset
from utils.training_utils import *
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan, arch_ops
import logging
from tensorflow import keras
import tensorflow_datasets as tfds
from network.resnet import ImagenetModel
import h5py as h5
from tqdm import tqdm
import numpy as np
from utils import np_dataset


def compute_loss(train_step, data, strategy):
    e_loss = strategy.experimental_run_v2(train_step, data)
    mean_e_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, e_loss, axis=None)
    return mean_e_losses


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()
    print('Loading Imagenet2012 dataset...')
    dataset = np_dataset.build_np_dataset(root=config.h5root, gpu_nums=config.gpu_nums, load_in_mem=config.load_in_mem,
                                          load_num=config.load_num)
    dataset = strategy.experimental_distribute_dataset(dataset)
    dataset = dataset.make_initializable_iterator()
    with strategy.scope():
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        # dataset = get_dataset(name=config.dataset,
        #                       seed=config.seed).train_input_fn(params=data_iter_params)
        # dataset = strategy.experimental_distribute_dataset(dataset)
        # data_iter = dataset.make_initializable_iterator()
        print("Constructing networks...")
        Encoder = ImagenetModel(resnet_size=50, num_classes=120, name='vgg_alter')
        learning_rate = tf.train.exponential_decay(0.0001, global_step, 150000 / config.gpu_nums,
                                                   0.8, staircase=False)
        E_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='e_opt', beta1=config.beta1)

        print("Building tensorflow graph...")

        def train_step(image, W):
            E_w = Encoder(image, training=True)
            with tf.variable_scope('recon_loss'):
                recon_loss_pixel = tf.reduce_mean(tf.square(E_w - W))
                e_loss = recon_loss_pixel

            add_global = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([add_global] + update_ops):
                E_opt = E_solver.minimize(e_loss, var_list=Encoder.trainable_variables)
                with tf.control_dependencies([E_opt]):
                    return tf.identity(e_loss)
        e_loss = compute_loss(train_step, dataset.get_next(), strategy)
        print('Building init module...')
        with tf.init_scope():
            init = [tf.global_variables_initializer(), dataset.initializer]
            saver_e = tf.train.Saver(Encoder.trainable_variables, restore_sequentially=True)
        print("Start training...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            # This step will add op into graph, so we moved it before freeze
            fixed_img, _ = sess.run(dataset.get_next())
            print(fixed_img.shape)
            print('Saving fixed fake image to dir %s... ' % (config.model_dir + '/reals.png'))
            save_image_grid(fixed_img, filename=config.model_dir + '/reals.png')
            if config.finalize:
                sess.graph.finalize()
            timer.update()
            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)

            print("Start iterations...")
            for iteration in range(config.total_step):
                e_loss_, lr_ = sess.run([e_loss, learning_rate])
                if iteration % config.print_loss_per_steps == 0:
                    timer.update()
                    print("step %d, e_loss %f, learning_rate % f, consuming time %s" %
                          (iteration, e_loss_, lr_, timer.runing_time_format))
                if iteration % config.save_per_steps == 0:
                    saver_e.save(sess, save_path=config.model_dir + '/vgg.ckpt',
                                 global_step=iteration, write_meta_graph=False)


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

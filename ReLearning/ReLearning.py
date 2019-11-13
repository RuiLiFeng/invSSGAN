from datasets import get_dataset
from utils.training_utils import *
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan, arch_ops
import logging
from network.resnet import ImagenetModel
import tensorflow_datasets as tfds
import h5py as h5
from tqdm import tqdm
from utils import np_dataset as npdt
from ReLearning import datasets
import numpy as np


EPS = 1e-15


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()
    print('Loading Imagenet2012 dataset...')
    # dataset = load_from_h5(root=config.h5root, batch_size=config.batch_size)
    dataset, _, fixed_img = datasets.build_data_input_pipeline_from_hdf5(
        root=config.h5root, batch_size=config.batch_size, gpu_nums=config.gpu_nums,
        load_in_mem=config.load_in_mem, labeled_per_class=config.labeled_per_class, save_index_dir=config.model_dir)
    dataset = strategy.experimental_distribute_dataset(dataset)
    dataset = dataset.make_initializable_iterator()
    eval_dset = datasets.build_eval_dset(config.eval_h5root, batch_size=config.batch_size, gpu_nums=config.gpu_nums)
    eval_dset = strategy.experimental_distribute_dataset(eval_dset)
    eval_dset = eval_dset.make_initializable_iterator()
    with strategy.scope():
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        print("Constructing networks...")
        fixed_x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        Encoder = ImagenetModel(resnet_size=50, num_classes=None, name='vgg_alter')
        Dense = tf.layers.Dense(1000, name='Final_dense')
        learning_rate = tf.train.exponential_decay(config.lr, global_step, 60000,
                                                   0.8, staircase=False)
        Dense_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, name='e_opt', beta2=config.beta2)
        print("Building tensorflow graph...")

        def train_step(image, label):
            w = Encoder(image, training=True)
            w = Dense(w)
            label = tf.one_hot(label, 1000)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(label, w)

            add_global = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([add_global] + update_ops):
                Dense_opt = Dense_solver.minimize(loss, var_list=Dense.trainable_variables)
                with tf.control_dependencies([Dense_opt]):
                    return tf.identity(loss)
        loss = strategy.experimental_run_v2(train_step, dataset.get_next())
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        print("Building eval module...")

        def eval_step(image, label):
            w = Encoder(image, training=True)
            w = Dense(w)
            p = tf.math.argmax(w)
            p = tf.cast(p, tf.int32)
            precise = tf.reduce_mean(tf.cast(tf.equal(p, label), tf.float32))
            return precise

        precise = strategy.experimental_run_v2(eval_step, eval_dset.get_next())
        precise = strategy.reduce(tf.distribute.ReduceOp.MEAN, precise, axis=None)
        print('Building init module...')
        with tf.init_scope():
            init = [tf.global_variables_initializer(), dataset.initializer, eval_dset.initializer]
            saver_e = tf.train.Saver(Encoder.trainable_variables, restore_sequentially=True)
            saver_dense = tf.train.Saver(Dense.trainable_variables, restore_sequentially=True)
        print("Start training...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            print("Restore Encoder...")
            saver_e.restore(sess, config.restore_v_dir)
            timer.update()
            print("Completing all work, iteration now start, consuming %s " % timer.runing_time_format)
            print("Start iterations...")
            for iteration in range(config.total_step):
                loss_, lr_ = sess.run([loss, learning_rate])
                if iteration % config.print_loss_per_steps == 0:
                    timer.update()
                    print("step %d, loss %f, learning_rate % f, consuming time %s" %
                          (iteration, loss_, lr_, timer.runing_time_format))
                if iteration % config.eval_per_steps == 0:
                    timer.update()
                    print('Starting eval...')
                    precise_ = 0.0
                    eval_iters = 50000 // config.batch_size
                    for _ in range(2 * eval_iters):
                        precise_ += sess.run(precise)
                    precise_ = precise_ / 2 * eval_iters
                    timer.update()
                    print('Eval consuming time %s' % timer.duration_format)
                    print('step %d, precision %f in eval dataset of length %d' %
                          (iteration, precise_, 1000 * config.batch_size))
                if iteration % config.save_per_steps == 0:
                    saver_dense.save(sess, save_path=config.model_dir + '/dense.ckpt',
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

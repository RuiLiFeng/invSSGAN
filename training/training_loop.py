from datasets import get_dataset
from utils.training_utils import *
from network import SSGAN, loss_lib, penalty_lib, tfmetric, resnet_biggan, arch_ops, invert
import logging


def compute_loss(train_step, strategy, globa_step, z, data_iter):
    per_replica_G_losses, per_is, per_fid = strategy.experimental_run_v2(train_step, ('G', globa_step, z, data_iter.get_next()))
    per_replica_D_losses, _, _ = strategy.experimental_run_v2(train_step, ('D', globa_step, z, data_iter.get_next()))
    mean_G_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_G_losses, axis=None)
    mean_D_losses = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_D_losses, axis=None)
    mean_is = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_is, axis=None)
    mean_fid = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_fid, axis=None)
    return mean_G_losses, mean_D_losses, mean_is, mean_fid


def compute_eval(eval_step, strategy, z, data_iter):
    per_replica_is, per_replica_fid, per_replica_sample = strategy.experimental_run_v2(
        eval_step, (z, data_iter.get_next()))
    mean_is = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_is, axis=None)
    mean_fid = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_fid, axis=None)
    mean_sample = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_sample, axis=None)
    return mean_is, mean_fid, mean_sample


def training_loop(config: Config):
    timer = Timer()
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()
    data_iter_params = {"batch_size": config.batch_size, "seed": config.seed}
    with strategy.scope():
        # ema = tf.train.ExponentialMovingAverage(0.999)
        global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False,
                                      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        dataset = get_dataset(name=config.dataset,
                              seed=config.seed).train_input_fn(params=data_iter_params)
        dataset = strategy.experimental_distribute_dataset(dataset)
        data_iter = dataset.make_initializable_iterator()
        print("Constructing networks...")
        InvMap = invert.InvMap(latent_size=config.dim_z)
        Generator = resnet_biggan.Generator(image_shape=[128,128,3], embed_y=False,
                                            embed_z=True,
                                            batch_norm_fn=arch_ops.self_modulated_batch_norm,
                                            spectral_norm=True)
        Discriminator = resnet_biggan.Discriminator(spectral_norm=True, project_y=False)
        I_opt = tf.train.AdamOptimizer(learning_rate=0.0005, name='i_opt', beta1=0.0, beta2=0.999)
        G_opt = tf.train.AdamOptimizer(learning_rate=0.00001, name='g_opt', beta1=0.0, beta2=0.999)
        D_opt = tf.train.AdamOptimizer(learning_rate=0.00005, name='d_opt', beta1=0.0, beta2=0.999)
        train_z = tf.random.normal([config.batch_size // config.gpu_nums, config.dim_z],
                                   stddev=1.0, name='train_z')
        # eval_z = tf.random.uniform([config.batch_size // config.gpu_nums, config.dim_z],
        #                             minval=-1.0, maxval=1.0, name='eval_z')
        # eval_z = tf.placeholder(tf.float32, name='eval_z')
        fixed_sample_z = tf.placeholder(tf.float32, name='fixed_sample_z')

        print("Building tensorflow graph...")

        def train_step(training_who="G", step=None, z=None, data=None):
            img, labels = data
            w = InvMap(z)
            samples = Generator(z=w, y=None, is_training=True)
            d_real, d_real_logits, _ = Discriminator(x=img, y=None, is_training=True)
            d_fake, d_fake_logits, _ = Discriminator(x=samples, y=None, is_training=True)
            d_loss, _, _, g_loss = loss_lib.get_losses(d_real=d_real, d_fake=d_fake, d_real_logits=d_real_logits,
                                                       d_fake_logits=d_fake_logits)

            inception_score = tfmetric.call_metric(run_dir_root=config.run_dir,
                                                   name="is",
                                                   images=samples)
            fid = tfmetric.call_metric(run_dir_root=config.run_dir,
                                       name="fid",
                                       reals=img,
                                       fakes=samples)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if training_who == "G":
                    train_op = tf.group(
                        G_opt.minimize(g_loss, var_list=Generator.trainable_variables, global_step=step),
                        I_opt.minimize(g_loss, var_list=InvMap.trainable_variables, global_step=step))
                    # decay = config.ema_decay * tf.cast(
                    #     tf.greater_equal(step, config.ema_start_step), tf.float32)
                    # with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                    #     ema = tf.train.ExponentialMovingAverage(decay=decay)
                    #     with tf.control_dependencies([train_op]):
                    #         train_op = ema.apply(Generator.trainable_variables + InvMap.trainable_variables)

                    with tf.control_dependencies([train_op]):
                        return tf.identity(g_loss), inception_score, fid
                else:
                    train_op = D_opt.minimize(d_loss, var_list=Discriminator.trainable_variables, global_step=step)
                    with tf.control_dependencies([train_op]):
                        return tf.identity(d_loss), inception_score, fid

        # def eval_step(z, data=None):
        #     img, _ = data
        #     # with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        #     #     ema = tf.train.ExponentialMovingAverage(decay=0.999)
        #     #     ema.apply(Generator.trainable_variables + InvMap.trainable_variables)
        #     #
        #     # def ema_getter(getter, name, *args, **kwargs):
        #     #     var = getter(name, *args, **kwargs)
        #     #     ema_var = ema.average(var)
        #     #     if ema_var is None:
        #     #         var_names_without_ema = {"u_var", "accu_mean", "accu_variance",
        #     #                                  "accu_counter", "update_accus"}
        #     #         if name.split("/")[-1] not in var_names_without_ema:
        #     #             logging.warning("Could not find EMA variable for %s.", name)
        #     #         return var
        #     #     return ema_var
        #     # with tf.variable_scope("", values=[z, img], reuse=tf.AUTO_REUSE,
        #     #                        custom_getter=ema_getter):
        #     w = InvMap(z)
        #     sampled = Generator(z=w, y=None, is_training=False)
        #     inception_score = tfmetric.call_metric(run_dir_root=config.run_dir,
        #                                            name="is",
        #                                            images=sampled)
        #     fid = tfmetric.call_metric(run_dir_root=config.run_dir,
        #                                name="fid",
        #                                reals=img,
        #                                fakes=sampled)
        #     return inception_score, fid, sampled

        g_loss, d_loss, IS, FID = compute_loss(train_step, strategy, global_step, train_z, data_iter)
        print("Building eval module...")
        with tf.init_scope():
            # IS, FID, eval_sample = compute_eval(eval_step, strategy, eval_z, data_iter)
            fixed_sample_w = InvMap(fixed_sample_z)
            eval_sample = Generator(z=fixed_sample_w, y=None, is_training=False)
        print('Building init module...')
        with tf.init_scope():
            init = [tf.global_variables_initializer(), data_iter.initializer]
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
            fixed_z = np.random.uniform(low=-1.0, high=1.0,
                                        size=[config.batch_size * 2 // config.gpu_nums, config.dim_z])
            print("Restore generator and discriminator...")
            saver_g.restore(sess, '/ghome/fengrl/gen_ckpt/gen-0')
            saver_d.restore(sess, '/ghome/fengrl/disc_ckpt/disc-0')
            print("Start iterations...")
            for iteration in range(config.total_step):
                for D_repeat in range(config.disc_iter):
                    D_loss = sess.run(d_loss)
                G_loss = sess.run(g_loss)
                if iteration % config.print_loss_per_steps == 0:
                    print("step %d, G_loss %f, D_loss %f" % (iteration, G_loss, D_loss))
                if iteration % config.eval_per_steps == 0:
                    timer.update()
                    fixed_sample = sess.run(eval_sample, {fixed_sample_z: fixed_z})
                    save_image_grid(fixed_sample, filename=config.model_dir + '/fakes%06d.png' % iteration)
                    is_eval, fid_eval = sess.run([IS, FID])
                    print("Time %s, fid %f, inception_score %f , G_loss %f, D_loss %f, step %d" %
                          (timer.runing_time, fid_eval, is_eval, G_loss, D_loss, iteration))
                if iteration % config.save_per_steps == 0:
                    saver_g.save(sess, save_path=config.model_dir + '/gen.ckpt',
                                 global_step=iteration, write_meta_graph=False)
                    saver_d.save(sess, save_path=config.model_dir + '/disc.ckpt',
                                 global_step=iteration,write_meta_graph=False)








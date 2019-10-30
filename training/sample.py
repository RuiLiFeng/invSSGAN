from utils.training_utils import *
import logging
from network.SSGAN import Generator
import tensorflow as tf
from network import arch_ops
import h5py as h5
from tqdm import tqdm


CHUNK_SIZE = 500


def training_loop(config: Config):
    print("Start task {}".format(config.task_name))
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        print("Constructing networks...")
        G = Generator(config.ssgan_dir, trainable=False)
        z = tf.placeholder(tf.float32)
        print("Building sample pipeline...")
        sample = G(z)
        embed = [v for v in tf.global_variables() if 'gen_module/generator/embed_z/kernel' in v.name]
        print('Find %d embeding kernel' % len(embed))
        w = tf.matmul(z, embed[0])
        sample_dict = {'img': sample, 'z': z, 'w': w}
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            sample_num = 200 * 10000 // config.batch_size
            np.random.seed(config.seed)
            for step in tqdm(range(sample_num)):
                z_ = np.random.normal(size=[config.batch_size, 120])
                out_dict = sess.run(sample_dict, {z: z_})
                if step == 0:
                    save_image_grid(out_dict['img'], filename=config.model_dir + '/fakes.png')
                    with h5.File(config.write_h5_dir + '/SSGAN128.hdf5', 'w') as f:
                        dsets = []
                        for key in out_dict:
                            value = out_dict[key]
                            dsets.append(
                                f.create_dataset(key, value.shape, value.dtype,
                                                 maxshape=((sample_num * config.batch_size,) + value.shape[1:]),
                                                 chunks=(CHUNK_SIZE,) + value.shape[1:],
                                                 compression=None))
                            dsets[-1][...] = value
                else:
                    with h5.File(config.write_h5_dir + '/SSGAN128.hdf5', 'a') as f:
                        for key in out_dict:
                            value = out_dict[key]
                            f[key].resize(f[key].shape[0] + value.shape[0], axis=0)
                            f[key][-value.shape[0]:] = value




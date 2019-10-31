import numpy as np
import h5py as h5
import tensorflow as tf


class ssgan_sample(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/SSGAN128.hdf5', batch_size=256):
        print('Loading data root %s into memory...' % root)
        self.file = h5.File(root, 'r')
        self.img = self.file['img']
        with h5.File(root.replace('SSGAN128', 'wema'), 'r') as f:
            self.w = f['w'][:]
        self.num_imgs = len(self.file['z'])
        self.index = np.arange(self.num_imgs)
        self.batch_size = batch_size

    def gen(self):
        for i in self.index:
            yield self.img[i], self.w[i]

    def __len__(self):
        return self.num_imgs


def parser_fn(img, w):
    return tf.cast(tf.transpose(img, [1, 2, 0]), tf.float32) / 255.0, w


def build_np_dataset(root, batch_size, gpu_nums, dset_name='ssgan_sample'):
    np_dset = {'ssgan_sample': ssgan_sample}[dset_name]
    h5_dset = np_dset(root, batch_size)
    dset = tf.data.Dataset.from_generator(h5_dset.gen, (tf.float32, tf.float32),
                                          output_shapes=([3, 128, 128], [120]))
    print('Making tensorflow dataset with length %d' % len(h5_dset))
    dset = dset.map(map_func=parser_fn, num_parallel_calls=3 * gpu_nums).shuffle(len(h5_dset)).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset
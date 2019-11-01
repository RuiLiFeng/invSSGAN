import numpy as np
import h5py as h5
import tensorflow as tf


class ssgan_sample(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/SSGAN128.hdf5', load_num=None, load_in_mem=False):
        print('Loading data root %s into memory...' % root)
        with h5.File(root.replace('SSGAN128', 'wema'), 'r') as f:
            self.w = f['w'][:]
        self.num_imgs = len(self.w) if load_num is None else load_num
        self.w = self.w[:self.num_imgs]
        self.load_in_mem = load_in_mem
        if self.load_in_mem:
            with h5.File(root, 'r') as f:
                self.img = f['img'][: self.num_imgs]
        self.root = root
        self.index = np.arange(self.num_imgs)

    def gen_from_mem(self):
        for i in self.index:
            yield self.img[i], self.w[i]

    def gen_from_file(self):
        with h5.File(self.root, 'r') as f:
            for i in self.index:
                yield f['img'][i], self.w[i]

    def __len__(self):
        return self.num_imgs


def parser_fn(img, w):
    return tf.cast(tf.transpose(img, [1, 2, 0]), tf.float32) / 255.0, w


def build_np_dataset(root, batch_size, gpu_nums, dset_name='ssgan_sample', load_in_mem=False, load_num=None,
                     shuffle_buffer_size=10000):
    np_dset = {'ssgan_sample': ssgan_sample}[dset_name]
    h5_dset = np_dset(root, load_num, load_in_mem)
    gen = h5_dset.gen_from_mem if load_in_mem else h5_dset.gen_from_file
    dset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32),
                                          output_shapes=([128, 128, 3], [120]))
    print('Making tensorflow dataset with length %d' % len(h5_dset))
    dset = dset.interleave()
    dset = dset.shuffle(shuffle_buffer_size).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset
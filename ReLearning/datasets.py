import numpy as np
import h5py as h5
from tqdm import tqdm
import tensorflow as tf


class Fdset_np(object):
    def __init__(self, root='/gpub/temp/imagenet2012/hdf5/ILSVRC128.hdf5', load_in_mem=True,
                 sample_batch_size=256, labeled_per_class=None, save_index_dir=None):
        """
        np dset for few shot learning.
        :param root:
        :param load_in_mem:
        :param sample_batch_size:
        :param labeled_per_class:
        :param save_index_dir:
        """
        self.root = root
        self.load_in_mem = load_in_mem
        self.sample_batch_size = sample_batch_size
        self.labeled_per_class = labeled_per_class
        self.save_index_dir = save_index_dir

        with h5.File(root, 'r') as f:
            self.label = f['labels'][1:]
        self.num_imgs = len(self.label)
        self.labeled_index, self.unlabeled_index = make_index(labeled_per_class, self.label)
        np.save(save_index_dir + '/labeled_index.npy', np.array(self.labeled_index))
        np.save(save_index_dir + '/unlabeled_index.npy', np.array(self.unlabeled_index))

        if self.load_in_mem:
            print('Load dataset %s into memory...' % root)
            with h5.File(root, 'r') as f:
                self.img = f['imgs'][1:]

        self.labeled_shuffle_index = np.arange(len(self.labeled_index))
        self.unlabeled_shuffle_index = np.arange(len(self.unlabeled_index))
        np.random.shuffle(self.labeled_shuffle_index)
        np.random.shuffle(self.unlabeled_shuffle_index)

    def gen_labeled_from_file(self):
        for i in self.labeled_shuffle_index:
            index = self.labeled_index[i]
            with h5.File(self.root, 'r') as f:
                yield f['imgs'][index + 1], f['labels'][index + 1]

    def gen_labeled_from_mem(self):
        for i in self.labeled_shuffle_index:
            index = self.labeled_index[i]
            yield self.img[index], self.label[index]

    def gen_unlabeled_from_file(self):
        for i in self.unlabeled_shuffle_index:
            index = self.unlabeled_index[i]
            with h5.File(self.root, 'r') as f:
                yield f['imgs'][index + 1], f['labels'][index + 1]

    def gen_unlabeled_from_mem(self):
        for i in self.unlabeled_shuffle_index:
            index = self.unlabeled_index[i]
            yield self.img[index], self.label[index]

    def fixed_sample_np(self):
        index = np.random.randint(0, self.num_imgs, self.sample_batch_size)
        index.sort()
        if self.load_in_mem:
            return self.img[list(index)].transpose([0, 2, 3, 1]) / 255.0
        else:
            with h5.File(self.root, 'r') as f:
                return f['imgs'][list(index + 1)].transpose([0, 2, 3, 1]) / 255.0

    def __len__(self):
        return self.num_imgs

    @property
    def labeled_num(self):
        return len(self.labeled_index)

    @property
    def unlabeled_num(self):
        return len(self.unlabeled_index)


class np_dset(object):
    def __init__(self, root):
        print('Loading dataset in %s into memory...' % root)
        with h5.File(root, 'r') as f:
            self.img = f['imgs'][:]
            self.label = f['labels'][:]
        self.num_imgs = len(self.label)
        self.index = np.arange(self.num_imgs)
        np.random.shuffle(self.index)

    def gen(self):
        for i in self.index:
            yield self.img[i], self.label[i]

    def __len__(self):
        return self.num_imgs


def make_index(labeled_per_class, labels):
    labeled_list = []
    unlabeled_list = []
    st = 0
    ed = 0
    print('Generating few shot index list with %d index per class' % labeled_per_class)
    for index in tqdm(range(len(labels))):
        if index > 0:
            if labels[index] != labels[index - 1]:
                print(index)
                ed = index
                all_index_in_this_class = np.arange(st, ed)
                np.random.shuffle(all_index_in_this_class)
                labeled_list = labeled_list + list(all_index_in_this_class[:labeled_per_class].sort())
                unlabeled_list = unlabeled_list + list(all_index_in_this_class[labeled_per_class:].sort())
                st = ed
    return labeled_list, unlabeled_list


def parser_fn(img, label):
    return tf.cast(tf.transpose(img, [1, 2, 0]), tf.float32) / 255.0, label


def build_data_input_pipeline_from_hdf5(root, batch_size, gpu_nums, load_in_mem,
                                        labeled_per_class, save_index_dir, shuffle_buffer_size=10000):
    h5set = Fdset_np(root, load_in_mem, batch_size / 4, labeled_per_class, save_index_dir)
    gen_labeled = h5set.gen_labeled_from_mem if load_in_mem else h5set.gen_labeled_from_file
    gen_unlabeled = h5set.gen_unlabeled_from_mem if load_in_mem else h5set.gen_unlabeled_from_file
    labeled_dset = tf.data.Dataset.from_generator(gen_labeled, (tf.float32, tf.int32),
                                                  output_shapes=([128, 128, 3], []))
    unlabeled_dset = tf.data.Dataset.from_generator(gen_unlabeled, (tf.float32, tf.int32),
                                                    output_shapes=([128, 128, 3], []))
    print('Making tensorflow datasets with length %d (labeled) and %d (unlabeled) from the'
          'whole datasets of length % d' % (h5set.labeled_num, h5set.unlabeled_num, len(h5set)))
    labeld_dset = labeled_dset.map(map_func=parser_fn,
                                   num_parallel_calls=2 * gpu_nums).shuffle(shuffle_buffer_size).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    unlabeld_dset = unlabeled_dset.map(map_func=parser_fn,
                                       num_parallel_calls=2 * gpu_nums).shuffle(shuffle_buffer_size).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    fixed_sample = h5set.fixed_sample_np()
    return labeld_dset, unlabeld_dset, fixed_sample


def build_eval_dset(root, gpu_nums, batch_size, shuffle_buffer_size=10000):
    npdset = np_dset(root)
    dset = tf.data.Dataset.from_generator(npdset.gen, (tf.float32, tf.int32), output_shapes=([128, 128, 3], []))
    print('Making dataset of length %d' % len(npdset))
    dset = dset.map(map_func=parser_fn, num_parallel_calls=2 * gpu_nums).shuffle(shuffle_buffer_size).batch(
        batch_size, drop_remainder=True).repeat().prefetch(tf.contrib.data.AUTOTUNE)
    return dset
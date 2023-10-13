import tensorflow as tf
import csv
from similarity_model_settings import *


def load_img(im1, im2, label):
    """
    Takes filenames and label and maps to final dataset sample
    :param im1: high quality image filepath
    :param im2: low quality image filepath
    :param label: label (Real=0/Fake=1) corresponding to the image pair
    :return: A tuple of images and labels to be used as inputs and another tuple to be used as targets (0 for feature distance)
            and label for HQ and LQ output).
    """
    im1 = tf.io.read_file(im1)
    im1 = tf.io.decode_png(im1)
    im1 = tf.cast(im1, tf.float32)
    im1 = tf.image.resize(im1, IMAGE_SIZE, method='bilinear')

    im2 = tf.io.read_file(im2)
    im2 = tf.io.decode_png(im2)
    im2 = tf.cast(im2, tf.float32)
    im2 = tf.image.resize(im2, IMAGE_SIZE, method='bilinear')
    return (im1, im2), \
           (tf.constant(0),
            tf.one_hot(tf.strings.to_number(label, out_type=tf.dtypes.int32), 2),
            tf.one_hot(tf.strings.to_number(label, out_type=tf.dtypes.int32), 2))


def load_names_labels(setname):
    """
    Constructs a dataset of image filepaths and labels, corresponding to the setname given (train/validation)
    :param setname: string name of set
    :return: a dataset containing filepaths and labels
    """
    high = []
    low = []
    labels = []
    with open("../../indexes/sim_RAW-HQ_" + setname + "_matched_files.csv", 'r', newline='', encoding='utf-8-sig') as f:
        r = csv.reader(f)
        for row in r:
            high.append(row[0])
            low.append(row[1])
            labels.append(row[2])
    return tf.data.Dataset.from_tensor_slices((high, low, labels))


def load_set(setname="train"):
    """
    Takes string name of dataset (train/validation) to be constructed.
    First constructs a dataset of filename pairs and corresponding labels; shuffles them, maps them to final dataset,
    batches and returns dataset.
    :param setname: string name of set to be constructed
    :return: the dataset
    """
    names_labels_ds = load_names_labels(setname)
    names_labels_ds = names_labels_ds.shuffle(buffer_size=100000)
    imgs_label_ds = names_labels_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    imgs_label_ds = imgs_label_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    return imgs_label_ds.prefetch(tf.data.AUTOTUNE)

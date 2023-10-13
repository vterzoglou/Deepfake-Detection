import tensorflow as tf
import csv
from adversarial_model_settings import *

qualities = ["LQ","HQ","RAW"]


def load_img(image, auth_label, qual_label):
    """
    Takes filenames and label and maps to final dataset sample
    :param image: high quality image filepath
    :param auth_label: authenticity label (Real=0/Fake=1) corresponding to the image
    :param qual_label: quality label (LQ=0/HQ=1) corresponding to the image
    :return: A tuple of the image (to be used as input) and its labels (to be used as output targets)
    """
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method='bilinear')

    return image, \
           (tf.one_hot(tf.strings.to_number(auth_label, out_type=tf.dtypes.int32), 2),
            tf.one_hot(tf.strings.to_number(qual_label, out_type=tf.dtypes.int32), 2))


def load_names_labels(setname, qual_idx=0):
    """
    Constructs a dataset of image filepaths and labels, corresponding to the setname given (train/validation)
    :param qual_idx: index of quality (LQ=0, HQ=1, RAW=2) to be used
    :param setname: string name of set
    :return: a dataset containing filepaths and labels
    """
    image_files = []
    auth_labels = []
    qual_labels = []
    with open("../../indexes/" + qualities[qual_idx] + "_" + setname + "_files.csv", 'r', newline='',encoding='utf-8-sig') as f:
        r = csv.reader(f)
        for row in r:
            image_files.append(row[0])
            auth_labels.append(row[1])
            qual_labels.append(row[2])
    return tf.data.Dataset.from_tensor_slices((image_files, auth_labels, qual_labels))


def load_set(setname="train"):
    """
    Takes string name of dataset (train/validation) to be constructed.
    First constructs a dataset of filename pairs and corresponding labels; shuffles them, maps them to final dataset,
    batches and returns dataset.
    :param setname: string name of set to be constructed
    :return: the dataset
    """

    # LQ + HQ samples are used by default.
    names_labels_ds_lq = load_names_labels(setname, 0)
    names_labels_ds_hq = load_names_labels(setname, 1)
    names_labels_ds = names_labels_ds_lq.concatenate(names_labels_ds_hq)
    if setname != "test":
        names_labels_ds = names_labels_ds.shuffle(buffer_size=155000)
    imgs_label_ds = names_labels_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    imgs_label_ds = imgs_label_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    return imgs_label_ds.prefetch(tf.data.AUTOTUNE)

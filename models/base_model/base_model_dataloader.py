import tensorflow as tf
import csv
from base_model_settings import *


qualities = ["LQ","HQ","RAW"]


def load_img(img, label):
    """
    Takes filename and label and maps to final dataset sample
    :param img: image filepath
    :param label: label (Real=0/Fake=1) corresponding to the image
    :return: A tuple of the image and its corresponding authenticity label
    """
    img = tf.io.read_file(img)
    img = tf.io.decode_png(img)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE, method='bilinear')

    return img, tf.one_hot(tf.strings.to_number(label, out_type=tf.dtypes.int32), 2)


def load_names_labels_qual(setname, qual_idx):
    """
    Constructs a dataset of image filepaths and labels, corresponding to the setname given (train/validation)
    :param qual_idx: index of quality (LQ=0, HQ=1, RAW=2) to be used
    :param setname: string name of set
    :return: a dataset containing filepaths and labels
    """
    image_files = []
    auth_labels = []
    with open("../../indexes/" + qualities[qual_idx] + "_" + setname + "_files.csv", 'r', newline='',encoding='utf-8-sig') as f:
        r = csv.reader(f)
        for row in r:
            image_files.append(row[0])
            auth_labels.append(row[1])
    return tf.data.Dataset.from_tensor_slices((image_files, auth_labels))


def load_set(setname="train", qual_idx=0):
    """
    Takes string name of dataset (train/validation) to be constructed.
    First constructs a dataset of filename pairs and corresponding labels; shuffles them, maps them to final dataset,
    batches and returns dataset.
    :param qual_idx: int representing the quality of data to be used. 0:lq, 1:hq, 2:raw, 3:raw+hq
    :param setname: string name of set to be constructed
    :return: the dataset
    """
    if qual_idx != 3:
        names_labels_ds = load_names_labels_qual(setname, qual_idx)
    else:
        names_labels_ds_hq = load_names_labels_qual(setname, 1)
        names_labels_ds_raw = load_names_labels_qual(setname, 2)
        names_labels_ds = names_labels_ds_hq.concatenate(names_labels_ds_raw)

    # Every call to the Dataset will shuffle the samples, on testing this will mess up labels, so dont shuffle
    if setname != "test":
        names_labels_ds = names_labels_ds.shuffle(buffer_size=155000)
    imgs_label_ds = names_labels_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    imgs_label_ds = imgs_label_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True)
    return imgs_label_ds.prefetch(tf.data.AUTOTUNE)
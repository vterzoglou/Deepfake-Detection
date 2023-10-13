import tensorflow as tf
import numpy as np
from similarity_model_dataloader import *
from ..base_model.train_base_model import create_generator, create_classifier, unfreeze_model, test_model
from numpy.random import seed
import json
from glob import glob
import os
from datetime import datetime
from functools import partial


tf.random.set_seed(123)
seed(123)


# TF class weights dont work for multi-output models, so a custom solution is used.
def weighted_cce(y_true, y_pred , weights):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    weights = tf.convert_to_tensor(weights)
    assert weights.shape == y_true.shape[1]

    out_batch_category = weights * y_true * tf.math.log(y_pred)
    assert out_batch_category.shape == y_true.shape
    out_batch_level = -tf.reduce_sum(out_batch_category, axis=1)
    assert out_batch_level.shape == y_true.shape[0]
    out = tf.reduce_mean(out_batch_level)
    return out


def create_callbacks(stamp):
    # val_Classifier-layer_balanced_accuracy -> HQ validation balanced accuracy
    # val_Classifier-layer_1_balanced_accuracy -> LQ    >>        >>      >>

    # Checkpoint callback
    checkpoint_filepath = CHECKPOINTS_FOLDER+stamp+"_checkpoint-{epoch:04d}.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_Classifier-layer_1_balanced_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)
    return model_checkpoint_callback


def create_model():
    # Input layers
    img1 = tf.keras.Input(shape=IMAGE_SIZE+(3,), name="img1-layer")
    img2 = tf.keras.Input(shape=IMAGE_SIZE+(3,), name="img2-layer")

    # Generator and output features
    G = create_generator()
    feat1 = G(img1)
    feat2 = G(img2)

    # Norm of feature distance is used as an output
    feat_out = tf.norm(feat1-feat2, axis=-1, name='feat_out')

    # Classifier and final output
    C = create_classifier()
    class1 = C(feat1)
    class2 = C(feat2)

    model = tf.keras.Model(inputs=[img1, img2], outputs=[feat_out, class1, class2], name="similarity-model")
    return model


def sub_train(model, class_weights, learning_rate, train_ds, val_ds, loss_weights=[1, 1, 1], n_part=1):
    """
    Function that implements a part (of two) of the training process


    :param model: tf.keras.model to be trained
    :param class_weights: dictionary of weights to assign to each class (0:real/1:fake),
    considering the imbalance of the dataset
    :param learning_rate: the learning rate to be used for this part
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    :param loss_weights: weights applied when calculating total loss function from loss functions per model output
    :param n_part: integer specifying which of the two parts is taking place
    :return: tuple (hist, model) of the history of fit and the trained model
    """
    model.compile(loss=[tf.losses.MeanAbsoluteError(name="Similarity_Loss"),
                        partial(weighted_cce, weights=class_weights),
                        partial(weighted_cce, weights=class_weights)],
                  metrics=[None, METRICS, METRICS],
                  loss_weights=loss_weights,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    callbacks = create_callbacks(stamp=timestamp)

    # Write a simple log
    with open(CHECKPOINTS_FOLDER + "log.txt", "a") as f:
        f.write(timestamp + f": Similarity Model training on HQ+RAW, part {n_part}, "
                            f"LR: {LEARNING_RATES[n_part-1]}, loss_weights: {loss_weights}, {BLOCKS_USED} blocks used"
                            f", BATCH_SIZE={BATCH_SIZE}, IMAGE_SIZE:{IMAGE_SIZE}\n")

    # Fit and dump history in a json
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS[n_part-1], callbacks=callbacks)
    json.dump(hist.history, open(CHECKPOINTS_FOLDER+timestamp+"_hist.json", 'w'))

    # Restore model to best (latest) stored checkpoint, delete other checkpoints
    list_of_files = glob(f'{CHECKPOINTS_FOLDER}{timestamp}_checkpoint*')
    latest_file = max(list_of_files, key=os.path.getctime)
    model.load_weights(latest_file)
    for f in list_of_files:
        if f != latest_file:
            os.remove(f)
    return hist, model


def train_model(loss_weights=[1, 1, 1]):
    """
    Function that trains the similarity model
    :param loss_weights: weights applied when calculating total loss function from loss functions per model output
    :return: tuple (model, hist1, hist2) of the trained model
    and the histories on the two training parts
    """

    # Load train, validation sets
    train_ds = load_set()
    val_ds = load_set("validation")
    total_samples = BATCH_SIZE * len(train_ds)

    # Set up class weights
    fake_samples = np.sum(np.array([tf.reduce_sum(tf.argmax(y[1],axis=1)).numpy() for x,y in train_ds]))
    real_samples = total_samples-fake_samples
    # class_weights[0]: weight for real samples, class_weights[1]: weight for fake samples
    class_weights = [(1/real_samples)*total_samples/2, (1/fake_samples)*total_samples/2]

    model = create_model()

    # Do a first training session with the feature generator frozen
    G = next((x for x in model.layers if x.name == "Generator-layer"), None)
    G.trainable = False
    hist1, model = sub_train(model, class_weights, LEARNING_RATES[0], train_ds, val_ds, loss_weights=loss_weights)

    # Second train session with some top feature generator layers unfrozen.
    # This is the main training part, the first training part can optionally be omitted.
    unfreeze_model(G, BLOCKS_USED)
    hist2, model = sub_train(model, class_weights, LEARNING_RATES[1], train_ds, val_ds, loss_weights=loss_weights, n_part=2)

    return model, hist1, hist2


def load_from_weights_file(file):
    model = create_model()
    model.load_weights(file)
    return model


if __name__ == '__main__':
    # Train similarity model
    trained_model, hist1, hist2 = train_model()

    # Load similarity model
    # model = load_from_weights_file("./saved_models/...")

    # Test similarity model
    G = next((x for x in trained_model.layers if x.name == "Generator-layer"), None)
    C = next((x for x in trained_model.layers if x.name == "Classifier-layer"), None)
    model = tf.keras.Sequential([G, C])
    accuracies = test_model(model)

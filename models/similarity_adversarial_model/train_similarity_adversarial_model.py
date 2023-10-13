import tensorflow as tf
import numpy as np
from similarity_adversarial_model_dataloader import *
from ..base_model.train_base_model import create_generator, create_classifier, unfreeze_model, test_model
from ..adversarial_model.gradient_reversal import *
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


def create_model(include_grl=False, generator=None, auth_classifier=None, quality_classifier=None, total_calls=None):
    img_high = tf.keras.Input(shape=(224, 224, 3), name="High-img-layer")
    img_low = tf.keras.Input(shape=(224, 224, 3), name="Low-img-layer")
    if generator is None:
        generator = create_generator()

    feature_high = generator(img_high)
    feature_low = generator(img_low)

    # Norm of feature distance is used as an output
    feat_out = tf.norm(feature_high - feature_low, axis=-1, name='feat_out')

    if include_grl:
        grl = GradientReversalLayer(total_calls)
        qual_input_high = grl(feature_high)
        qual_input_low = grl(feature_low)
    else:
        qual_input_high = feature_high
        qual_input_low = feature_low

    if auth_classifier is None:
        auth_classifier = create_classifier("Authenticity-Classifier")
    if quality_classifier is None:
        quality_classifier = create_classifier("Quality-Classifier")

    quality_out_high = quality_classifier(qual_input_high)
    quality_out_low = quality_classifier(qual_input_low)
    auth_out_high = auth_classifier(feature_high)
    auth_out_low = auth_classifier(feature_low)

    model = tf.keras.Model(inputs=[img_high,img_low],
                           outputs=[feat_out,quality_out_high, quality_out_low,auth_out_high,auth_out_low],
                           name="sim-adversarial-model")
    return model


def create_callbacks(stamp):

    # Checkpoint callback
    checkpoint_filepath = CHECKPOINTS_FOLDER+stamp+"_checkpoint-{epoch:04d}.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_Authenticity-Classifier_1_balanced_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)
    return model_checkpoint_callback


def sub_train(model, class_weights, learning_rate, train_ds, val_ds, loss_weights=[1, 1, 1, 1, 1], n_part=1):
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
                        tf.keras.losses.CategoricalCrossentropy(name="High-Quality-loss"),
                        tf.keras.losses.CategoricalCrossentropy(name="Low-Quality-loss"),
                        partial(weighted_cce, weights=class_weights),
                        partial(weighted_cce, weights=class_weights)],
                  metrics=[None, METRICS, METRICS, METRICS, METRICS],
                  loss_weights=loss_weights,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    callbacks = create_callbacks(stamp=timestamp)

    # Write a simple log
    with open(CHECKPOINTS_FOLDER + "log.txt", "a") as f:
        f.write(timestamp + f": Sim-Adversarial Model training on HQ+RAW, part {n_part}, "
                            f"LR: {LEARNING_RATES[n_part-1]}, lossweights={loss_weights}, BatchSize={BATCH_SIZE}, "
                            f"{BLOCKS_USED} blocks used, IMAGE_SIZE:{IMAGE_SIZE}\n")

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


def train_model(loss_weights=[1, 1, 1, 1, 1]):
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
    fake_samples = np.sum(np.array([tf.reduce_sum(tf.argmax(y[3],axis=1)).numpy() for x,y in train_ds]))
    real_samples = total_samples-fake_samples
    # class_weights[0]: weight for real samples, class_weights[1]: weight for fake samples
    class_weights = [(1/real_samples)*total_samples/2, (1/fake_samples)*total_samples/2]

    # In the first training phase no updates are done and no loss propagation is utilized on the feature extractor
    # (frozen), so the GRL is not included.
    model = create_model(include_grl=False)
    gen = next((x for x in model.layers if x.name == "Generator-layer"), None)
    auth = next((x for x in model.layers if x.name == "Authenticity-Classifier"), None)
    qual = next((x for x in model.layers if x.name == "Quality-Classifier"), None)

    # Do a first training session with the feature generator frozen
    gen.trainable = False
    hist1, model = sub_train(model, class_weights, LEARNING_RATES[0], train_ds, val_ds, loss_weights=loss_weights)
    unfreeze_model(gen, BLOCKS_USED)

    # Second train session with some top feature generator layers unfrozen.
    # This is the main training part, the first training part can optionally be omitted.
    total_calls = total_samples/BATCH_SIZE*EPOCHS[1]
    model = create_model(include_grl=True, generator=gen, auth_classifier=auth, quality_classifier=qual, total_calls=total_calls)
    hist2, model = sub_train(model, class_weights, LEARNING_RATES[1], train_ds, val_ds, loss_weights=loss_weights, n_part=2)

    return model, hist1, hist2


def load_from_weights_file(file):
    model = create_model()
    # might break if GRL not included while creating model...
    model.load_weights(file)
    return model


if __name__ == '__main__':
    # Train sim-adversarial model
    trained_model, hist1, hist2 = train_model()

    # Load sim-adversarial model
    trained_model = load_from_weights_file("./saved_models/sim_adv_model.hdf5")

    # Test sim-adversarial model
    G = next((x for x in trained_model.layers if x.name == "Generator-layer"), None)
    C = next((x for x in trained_model.layers if x.name == "Authenticity-Classifier"), None)
    model = tf.keras.Sequential([G, C])
    accuracies = test_model(model)

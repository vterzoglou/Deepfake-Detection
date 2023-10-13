import tensorflow as tf
import numpy as np
from base_model_dataloader import *
from keras.applications.efficientnet import EfficientNetB0
from numpy.random import seed
import json
from glob import glob
import os
from datetime import datetime

tf.random.set_seed(123)
seed(123)


def unfreeze_model(model, numblocks):
    """
    :param model: model to unfreeze
    :param numblocks: number of blocks to unfreeze from top, out of 7
    :return: None
    """
    num_layers = len(model.layers)
    model.trainable = True

    first_trainable_num = next((i for i,x in enumerate(model.layers) if x.name.startswith("block"+str(7-numblocks+1))), None)
    for layer_idx in range(num_layers):
        layer = model.layers[layer_idx]

        if numblocks < 7 and layer_idx < first_trainable_num:
            layer.trainable = False
        else:
            # BatchNorm layers should remain frozen
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def create_callbacks(stamp):
    # Checkpoint callback
    checkpoint_filepath = CHECKPOINTS_FOLDER+stamp+"_checkpoint-{epoch:04d}.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_balanced_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)
    return model_checkpoint_callback


def sub_train(model, class_weights, learning_rate, train_ds, val_ds, n_part=1):
    """
    Function that implements a part (of two) of the training process

    :param model: tf.keras.model to be trained
    :param class_weights: dictionary of weights to assign to each class (0:real/1:fake),
    considering the imbalance of the dataset
    :param learning_rate: the learning rate to be used for this part
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    :param n_part: integer specifying which of the two parts is taking place
    :return: tuple (hist, model) of the history of fit and the trained model
    """

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=METRICS,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    callbacks = create_callbacks(stamp=timestamp)

    # Write a simple log
    with open(CHECKPOINTS_FOLDER + "log.txt", "a") as f:
        f.write(timestamp + f": Base Model training on quality: {QUALITIES[QUALITY_TRAINED]}, part {n_part}, "
                            f"LR: {LEARNING_RATES[n_part-1]}, "
                            f"{BLOCKS_USED} blocks used, BATCH_SIZE={BATCH_SIZE}, "
                            f"IMAGE_SIZE:{IMAGE_SIZE}\n")

    # Fit and dump history in a json
    hist = model.fit(train_ds, validation_data=val_ds, class_weight=class_weights, epochs=EPOCHS[n_part-1], callbacks=callbacks)
    json.dump(hist.history, open(CHECKPOINTS_FOLDER+timestamp+"_hist.json", 'w'))

    # Restore model to best (latest) stored checkpoint, delete other checkpoints
    list_of_files = glob(f'{CHECKPOINTS_FOLDER}{timestamp}_checkpoint*')
    latest_file = max(list_of_files, key=os.path.getctime)
    model.load_weights(latest_file)
    for f in list_of_files:
        if f != latest_file:
            os.remove(f)
    return hist, model


def create_generator():
    inputs = tf.keras.Input(shape=IMAGE_SIZE+(3,))
    efnet = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    # Reduce output features to vectors
    output = tf.keras.layers.GlobalAveragePooling2D()(efnet.output)

    model = tf.keras.Model(inputs, output, name="Generator-layer")
    return model


def create_classifier(num_outputs=2):
    model = tf.keras.Sequential([
        tf.keras.Input(1280,),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_outputs, activation="softmax")
    ], name="Classifier-layer")
    return model


def create_model():
    """
    Function that creates a baseline model,
    consisting of a feature generator and an classifier
    :return: the model
    """
    # Input layer
    img = tf.keras.Input(shape=IMAGE_SIZE+(3,), name="img-layer")

    # Generator and output features
    G = create_generator()
    feature = G(img)

    # Classifier and final output
    C = create_classifier()
    auth_out = C(feature)

    model = tf.keras.Model(inputs=[img], outputs=[auth_out], name="base-model")
    return model


def train_model():
    """
    Function that trains the base model
    :return: tuple (model, hist1, hist2) of the trained model
    and the histories on the two training parts
    """

    # Load train, validation sets
    train_ds = load_set(qual_idx=QUALITY_TRAINED)
    val_ds = load_set("validation", qual_idx=QUALITY_TRAINED)

    # Set up class weights
    total_samples = BATCH_SIZE * len(train_ds)
    fake_samples = np.sum(np.array([tf.reduce_sum(tf.argmax(y,axis=1)).numpy() for x,y in train_ds]))
    real_samples = total_samples-fake_samples
    wf = (1/fake_samples) * (total_samples/2)
    wr = (1/real_samples) * (total_samples/2)
    class_weights = {1: wf, 0: wr}

    model = create_model()

    # Do a first training session with the feature generator frozen
    G = next((x for x in model.layers if x.name == "Generator-layer"), None)
    G.trainable = False
    hist1, model = sub_train(model, class_weights, LEARNING_RATES[0], train_ds, val_ds)

    # Second train session with some top feature generator layers unfrozen.
    # This is the main training part, the first training part can optionally be omitted.
    unfreeze_model(G, BLOCKS_USED)
    hist2, model = sub_train(model, class_weights, LEARNING_RATES[1], train_ds, val_ds, n_part=2)

    return model, hist1, hist2


def load_from_weights_file(file):
    model = create_model()
    model.load_weights(file)
    return model


def test_model(model):
    """
    Function that tests base model on the three qualities of frames extracted from FF++ dataset
    :param model: model to be testes
    :return: list of accuracies on lq, hq, raw testsets
    """
    testsets = [load_set('test', qual_idx) for qual_idx in range(len(qualities))]

    y_true_sem = [np.concatenate([y.numpy() for _, y in testset], axis=0) for testset in testsets]
    y_preds = [model.predict(testset) for testset in testsets]

    evaluations_semantic = [metrics.balanced_accuracy(y_t, y_p).numpy() for y_t, y_p in zip(y_true_sem, y_preds)]
    return evaluations_semantic


if __name__ == '__main__':

    # Train base model
    trained_model, hist1, hist2 = train_model()

    # Load base model
    model = load_from_weights_file("./saved_models/RAW")

    # Test model
    accuracies = test_model(model)
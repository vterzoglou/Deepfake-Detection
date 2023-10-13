import csv
import tensorflow as tf
import numpy as np
from frame_extraction import *
import pickle
from warnings import filterwarnings
from models.base_model import train_base_model
from tqdm import tqdm
import metrics
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib
matplotlib.use("Agg")

QUALITY_CLASSIFIERS = ['./saved_models/Centroid_classifier.pkl',
                       './saved_models/GNB_classifier.pkl']
CLASSIFIERS_NAMES = ["Centroid", "GNB"]
LQ_model_file = '../base_model/saved_models/LQ.hdf5'
HQ_model_file = '../base_model/saved_models/HQ.hdf5'
RAW_model_file = '../base_model/saved_models/RAW.hdf5'
IMAGE_SIZE = (224, 224)


def make_ds(vid, detector, quality_weights):
    video_samples = get_samples_from_vid_file(vid)

    # faces is a list of faces
    faces = detect_faces_in_samples(video_samples, detector)
    faces = [tf.image.resize(tf.convert_to_tensor(face), IMAGE_SIZE, method='bilinear') for face in faces]
    weights = [quality_weights for _ in faces]
    return [np.array(faces), np.array(weights)]


def predict_videos_quality(log_nbrs, classif_ind):
    log_nbrs = np.array(log_nbrs).reshape(-1, 1)
    with open(QUALITY_CLASSIFIERS[classif_ind], 'rb') as f:
        qual_classifier = pickle.load(f)

    if type(qual_classifier).__name__ == "NearestCentroid":
        preds = qual_classifier.predict(log_nbrs)
        tf_preds = tf.one_hot(preds, 3)
        return tf_preds

    preds = qual_classifier.predict_proba(log_nbrs)
    tf_preds = tf.convert_to_tensor(preds)
    return tf_preds


def create_submodels():
    # Load weights
    lq_model = train_base_model.create_model()
    train_base_model.unfreeze_model(lq_model, 4)
    lq_model.load_weights(LQ_model_file)

    hq_model = train_base_model.create_model()
    G_hq = next((x for x in hq_model.layers if x.name == "Generator-layer"), None)
    train_base_model.unfreeze_model(G_hq, 4)
    hq_model.load_weights(HQ_model_file)

    raw_model = train_base_model.create_model()
    G_raw = next((x for x in raw_model.layers if x.name == "Generator-layer"), None)
    train_base_model.unfreeze_model(G_raw, 4)
    raw_model.load_weights(RAW_model_file)

    lq_model._name = 'LQ-model'
    hq_model._name = 'HQ-model'
    raw_model._name = 'RAW-model'


    return lq_model, hq_model, raw_model


def make_model(classif_ind):
    lq_model, hq_model, raw_model = create_submodels()

    img = tf.keras.Input(shape=IMAGE_SIZE+(3,), name="Image_Layer")

    weights = tf.keras.Input(shape=(3,), name="Weights_layer")

    lq_out = lq_model(img)
    hq_out = hq_model(img)
    raw_out = raw_model(img)

    # stack along last axis, f_j.shape = [num of semantic cat, batch_size, num of domains]
    f_j = tf.stack([tf.transpose(lq_out), tf.transpose(hq_out), tf.transpose(raw_out)], -1)
    # weights.shape = [batch_size, num of domains]
    # elementwise multiplication over last 2 axes, sum over last axis
    z = tf.reduce_sum(tf.math.multiply(weights, f_j), -1)

    model = tf.keras.Model(inputs=[img, weights], outputs=[tf.transpose(z)])
    model._name = f'Ensemble + {CLASSIFIERS_NAMES[classif_ind]}'
    return model


def output_stats(model_name, videos_frames_preds):
    videos_preds = np.array([video_frames_preds.mean(axis=0).argmax() for video_frames_preds in videos_frames_preds])
    video_ba = metrics.balanced_accuracy(videos_labels, tf.one_hot(videos_preds, 2).numpy())

    video_frames = np.array([video_frames_preds.shape[0] for video_frames_preds in videos_frames_preds])
    frame_labels = tf.one_hot(np.repeat(videos_labels.argmax(axis=1), video_frames), 2)
    frame_ba = metrics.balanced_accuracy(frame_labels, np.concatenate(videos_frames_preds, axis=0))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    with open('./checkpoints/video_inferencing/logs.txt', 'a', newline='', encoding='utf-8-sig') as f:
        f.write(f"{timestamp}, Model:{model_name},\n"
                f"Video BAs:{video_ba.numpy()},\n"
                f"Frame BAs:{frame_ba.numpy()}, CRF:{CRF}\n\n")

    conf_matrix = confusion_matrix(videos_labels.argmax(axis=1), videos_preds)

    cm_disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Real', 'Fake'])
    disp = cm_disp.plot(cmap='Blues')
    matplotlib.pyplot.savefig(f"./checkpoints/video_inferencing/{timestamp}_c{CRF}_confmat.png")
    matplotlib.pyplot.savefig(f"./checkpoints/video_inferencing/{timestamp}_c{CRF}_confmat.svg")


if __name__ == '__main__':
    # Test compression level
    CRF = 23

    # classifier_index = 0 -> use Nearest Centroid Classifier
    # classifier_index = 1 -> use Gaussian Naive Bayes Classifier
    classifier_index = 0

    videos = []
    videos_labels = []
    log_nbrs = []
    testset_file = f'../../indexes/c{CRF}_video_index.csv'
    with open(testset_file, 'r', newline='', encoding='utf-8-sig') as f:
        r = csv.reader(f)
        for row in r:
            videos.append(row[0])
            videos_labels.append(row[1])
            log_nbrs.append(float(row[3]))
    videos_labels = np.array([tf.one_hot(int(y_t), 2) for y_t in videos_labels])

    face_detector = create_facial_detector()
    # video_qual_preds is a Tensor
    video_qual_preds = predict_videos_quality(log_nbrs, classifier_index)

    # Batch face detection may produce some warning (detecting different number of faces across batch), suppress it
    filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    model = make_model(classifier_index)

    videos_frames_preds = []
    for video_num, video in enumerate(tqdm(videos)):
        ds = make_ds(video, face_detector, video_qual_preds[video_num])
        videos_frames_preds.append(model(ds).numpy())

    model_name = model.name
    output_stats(model_name, videos_frames_preds)
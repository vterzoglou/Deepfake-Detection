import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
import csv

MARGIN = 1.3


def create_facial_detector():
    face_detector = MTCNN(
        keep_all=True,
        post_process=False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        select_largest=False,
        min_face_size=100,
        selection_method="probability",
        thresholds=[0.65, 0.75, 0.95],
        margin=0,
    )
    return face_detector


def expand_box(box):
    """
    Function that takes as input a bounding box
    and expands it by a factor of MARGIN
    :param box: np.array, the box to be expanded
    :return: np.array, box after expansion
    """
    l_x = box[2]-box[0]
    l_y = box[3]-box[1]
    c_x = (box[2]+box[0])/2
    c_y = (box[3]+box[1])/2

    boxnew = np.round(np.array([c_x, c_y, c_x, c_y]) + (np.array([-l_x, -l_y, l_x, l_y]) * MARGIN/2))
    return boxnew.astype(int)


def find_biggest_faces_expand(images_faces_boxes):
    """
    Function that selects the biggest (by area) facial bounding box
    for each frame and expands it by a factor of MARGIN

    :param images_faces_boxes: ndaray of shape: [images x faces x 4]
    containing bounding boxes for each face detected in each frame
    :return: a list containing the biggest box for each image
    """

    boxes = []
    for imagenum, image_faces_boxes in enumerate(images_faces_boxes):
        # If no face is detected, continue
        if image_faces_boxes is None:
            boxes.append(None)
            continue

        # Find the index of the face bounding box with the biggest area
        max_index = ((image_faces_boxes[:,3]-image_faces_boxes[:,1])*
                              (image_faces_boxes[:,2]-image_faces_boxes[:,0])).argmax()
        box = image_faces_boxes[max_index, :].squeeze()

        # Expand box by a factor of MARGIN
        boxes.append(expand_box(box))
    return boxes


def detect_faces_in_samples(samples, detector, incl_framenums=False):
    """
    Function that  takes as input an array of samples (frames),
    detects the faces present in each frame using a specified detector
    and crops each frame on the biggest area face.

    The bounding box is expanded by a factor of MARGIN (global),
    if no face is detected on a frame, the frame is not considered.

    :param samples: np.array containing the frames to detect faces from
    :param detector: model used to detect faces in each frame (works with MTCNN model from facenet_pytorch)
    :return: a list of frames cropped at the face area
    """


    # Detect faces in all frames
    # images_faces_boxes: [images x faces x 4] (ragged)
    images_faces_boxes, images_faces_probs = detector.detect(samples)

    #
    image_boxes = find_biggest_faces_expand(images_faces_boxes)
    faces = []

    xm, ym, _ = samples[0].shape
    frame_nums = []
    for img_num, image_box in enumerate(image_boxes):
        if image_box is None:
            continue

        faces.append(samples[img_num][max(image_box[1], 0):min(image_box[3], xm-1),
                             max(image_box[0], 0):min(image_box[2], ym-1), :])
        if incl_framenums:
            frame_nums.append(img_num)
    if not incl_framenums:
        return faces
    return faces, frame_nums


def get_samples_from_vid_file(filename, target_fps=1):

    """
    Function that takes as input a path of a video and extracts frames,
    sampling it at a specified fps rate

    :param filename: path to video
    :param target_fps: fps rate at which to sample the video
    :return: np.array of frames extracted
    """
    cap = cv2.VideoCapture(filename)
    source_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = int(cap.get(cv2.CAP_PROP_FPS))

    step_extract = int(np.ceil(source_fps/target_fps))

    frame_extract_ind = np.arange(0, source_frames_total, step_extract)
    samples = []
    for frame_num in range(source_frames_total):
        _, frame = cap.read()
        if frame_num not in frame_extract_ind:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        samples.append(frame)
    return np.array(samples)


if __name__ == '__main__':

    # Quality to extract frames for, 0:RAW, 23:HQ, 40:LQ
    CRF = 0
    sources = []

    # parse source video paths
    with open(f"./indexes/c{CRF}_video_index.csv",'r',newline='', encoding='utf-8-sig') as f:
        r = csv.reader(f)
        for row in r:
            sources.append(row[0])

    detector = create_facial_detector()
    for source in sources:
        video_samples = get_samples_from_vid_file(source)

        faces, frame_nums = detect_faces_in_samples(video_samples, detector)

        category, video_num = source.split("/")[-2:]
        video_num = video_num.split('.')[0]
        target = f"./dataset/ff++/{category}/c{CRF}/{video_num}/"

        # Write target frames
        for face, frame_num in zip(faces, frame_nums):
            target_face = target+f"{frame_num}.png"
            cv2.imwrite(face, target_face)

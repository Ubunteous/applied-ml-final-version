import numpy as np
# from keras.preprocessing import image # outdated
import keras.utils as image
import cv2
import dlib

import pandas as pd
# import csv, os
# from tools.landmarks import run_dlib_shape
# from tools.load_data import get_data_dir
from .load_data import get_data_dir
from tqdm import tqdm
from pathlib import Path

cwd = Path(__file__).resolve()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(cwd.parents[1] / "shape_predictor_68_face_landmarks.dat"))

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(data_dir, req_label, max_data):
    # inputs: folder and label, split between nb of training inputs and tests inputs starting from 0

    # landmark / points to return
    # outputs: landmarks and labels
    
    """
    This funtion extracts the landmarks features for all
    images in the appropriate folder.
    :return:
    landmark_features:  an array containing 0-68 landmark
    points for each image in which a face was detected
    label: an array containing a label for each image in
    which a face was detected
    """

    # place labels.csv in a dataframe
    images_dir = data_dir / "img"
    df = pd.read_csv(data_dir / "labels.csv", sep="\t")

    # get appropriate label and image indexes
    genders = df[req_label]
    df.rename(columns={'Unnamed: 0':"img_index"}, inplace=True)
    images_index = df["img_index"]

    target_size = None
    labels = None

    all_features = []
    all_labels = []

    # celeba = .jpg and cartoon = .png
    extension = ".jpg" if req_label in ["gender", "smiling"] else ".png"
    
    # training
    # we can just use a range since files are called n.jpg
    # for image_index in images_index:
    for image_index in tqdm(range(0, max_data)):
        # if image_index % 100 == 0:
        #     print(f"Image {image_index} / nb_images")
            
        # generate tf / keras images
        img_path = images_dir / (str(image_index) + extension)

        img = image.img_to_array(image.load_img(img_path,
                       target_size=target_size,
                       interpolation='bicubic'))

        # get features
        features, _ = run_dlib_shape(img)
        if features is not None:
            all_features.append(features)
            all_labels.append(genders[image_index])

    landmark_features = np.array(all_features)

    # for binary classification, we avoid -1 values
    if req_label in ["gender", "smiling"]: 
        # converts the -1 into 0, so male=0 and female=1
        gender_labels = (np.array(all_labels) + 1)/2
        return landmark_features, gender_labels

    # convert all multiclass cartoon labels to numpy
    return landmark_features, np.array(all_labels)

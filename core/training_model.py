from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
from numpy import load, expand_dims, asarray, savez_compressed
from keras.models import load_model
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image
import os
import pandas as pd

# import gc


# function for face detection with mtcnn


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert("RGB")
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# ONLY FACES OF ONE PERSON IS EXTRACTED


def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


# MAKING DATABASES OF ALL PEOPLE IN DATASET
# X STORES THE IMAGES AND Y STORES THE LABELS


def load_dataset(directory):
    X, y = list(), list()

    # THIS LOOP ITERATES OVER TRAIN AND VAL DIRECORY
    for subdir in os.listdir(directory):
        path = directory + subdir + "/"  # PATHS FOR INDIVIDUAL PERSON

        faces = load_faces(path)
        # CREATING LABELS
        labels = [subdir for _ in range(len(faces))]
        print(">loaded %d examples for class: %s" % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# get the face embedding for one face we use face_net_model
def get_embedding(face_net_model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype("float32")
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = face_net_model.predict(samples)

    return yhat[0]


def extract_faces(filename, required_size=(160, 160)):
    # load image from file
    face_array = []
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert("RGB")
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    for result in results:
        # extract the bounding box from the first face
        x1, y1, width, height = result["box"]
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array.append(asarray(image))

    return face_array


# from random import choice
# TAKING TWO INPUTS ONE IS SCREENCSHOT PIC AND OTHER IS DATE
def result(image, input_date):
    # LOAD FACES

    data = load("dataset for OAS_dummy.npz")
    testX_faces = image
    # load face embeddings
    data = load("face dataset embeddings_OAS_dummy.npz")
    trainX, trainy, testX, testy = (
        data["arr_0"],
        data["arr_1"],
        data["arr_2"],
        data["arr_3"],
    )

    # NORMALIZE  INPUT VECTOR

    in_encoder = Normalizer(norm="l2")
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    # LABEL ENCODING

    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    # FIT MODEL

    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(trainX, trainy)

    img_pixels = extract_faces(image)  # EXTRACTING FACE FROM INPUT IMAGE

    face_net_model = load_model("facenet_keras.h5")  # LOADING FACENET MODEL
    all_titles = []
    for test_img_pixels in img_pixels:
        face_pixels = test_img_pixels.astype("float32")

        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        tface_pixels = (face_pixels - mean) / std
        emb = get_embedding(face_net_model, tface_pixels)
        test_samples = np.expand_dims(emb, axis=0)

        test_yhat_class = svm_model.predict(test_samples)  # PREDICTING THE IMAGE
        test_yhat_prob = svm_model.predict_proba(test_samples)

        class_index = test_yhat_class[0]
        class_probability = test_yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(test_yhat_class)
        print("Predicted: %s (%.3f)" % (predict_names[0], class_probability))
        pyplot.imshow(tface_pixels)
        title = "%s (%.3f)" % (predict_names[0], class_probability)
        pyplot.title(title)
        # pyplot.show()
        all_titles.append(predict_names[0])  # ALL THE NAMES PREDICTED

    df = pd.read_excel("dummy sheet.xlsx")

    for title in all_titles:
        row = df.NAME[df.NAME == title].index.tolist()
        df.at[row[0], input_date] = "P"
    df[input_date] = df[input_date].fillna("A")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.to_excel("dummy sheet.xlsx")
    print(df)
    return df

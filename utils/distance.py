import tensorflow.keras.backend as K
import numpy as np


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    # sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def euclidean_distance_numpy(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    dist = np.linalg.norm(featsA - featsB)
    return dist

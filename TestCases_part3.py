import unittest
import pickle
import Kmeans_Students as km
import numpy as np
from Kmeans_Students import *
from utils import *
import json
import os
import KNN as k
from KNN import *
from skimage import io
import my_labeling as ml
import random


class TestCases(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        with open('./test/test_cases_knn.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)
            pass

    def test_retrieval_by_color(self):
        imatges = list()
        labels = list()
        colors = list()
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km.fit()
            labels.append(get_colors(km.centroids))
            imatges.append(input)
        # for i in range(0, random.randrange(1, 4)):
        #     colors.append(utils.colors[random.randrange(0, 11)])
        colors.append("White")
        colors.append("Red")
        out = ml.Retrieval_by_color(imatges, labels, colors)
        pass

    def test_retrieval_by_shape(self):
        imatges = list()
        labels = list()
        shapes = list()
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            labels.append(knn.predict(self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix]))
            imatges.append(knn.train_data)
        shapes.append(utils.shapes[random.randrange(0, 8)])
        out = ml.Retrieval_by_shape(imatges, labels, shapes)
        pass


if __name__ == '__main__':
    unittest.main()

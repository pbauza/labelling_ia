__authors__ = ['1491858', '1493406', '1493962']
__group__ = 'DL15'

import numpy as np
import copy
import utils
import math

class KMeans:

    def __init__(self, X, K=1, options=None):

        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()

    def _init_X(self, X):

        if len(X[0] != 3):
            X = X.reshape(len(X)*len(X[0]), 3)

        self.X = X.astype(float)

    def _init_options(self, options=None):
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):

        self.centroids = np.empty([self.K, 3], float)
        self.centroids[:] = np.nan
        self.old_centroids = np.empty([self.K, 3], float)
        self.old_centroids[:] = np.nan
        index_pixel = 0
        if self.options['km_init'].lower() == 'first':
            for pixel in self.X:
                if not any(np.equal(pixel, self.centroids).all(1)):
                    self.centroids[index_pixel] = pixel
                    index_pixel += 1
                    if index_pixel == self.K:
                        break
        elif self.options['km_init'].lower() == 'random': #MIRAR QUE NO ES REPETEIXIN ELS CENTROIDS
            self.centroids = np.random.rand(self.K, self.X.shape[1])
        #elif self.options['km_init'].lower() == 'custom':

    def get_labels(self):

        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)


    def get_centroids(self):

        self.old_centroids = np.copy(self.centroids[:])
        for index in range(0, len(self.centroids)):
            pixels_per_centroids = np.where(self.labels == index)[0]
            self.centroids[index] = self.X[pixels_per_centroids[:]].sum(0)/len(pixels_per_centroids)

    def converges(self):

        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):

        while self.converges() is False and self.num_iter != self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def withinClassDistance(self):

        dist = 0
        for i in range(0, len(self.centroids)):
            pixels_per_centroids = np.where(self.labels == i)[0]
            dist += np.sum((self.X[pixels_per_centroids[:]]-self.centroids[i])**2)
        return dist/len(self.X)

    def find_bestK(self, max_K):

        self.K = 2
        self._init_centroids()
        self.fit()
        aux = self.withinClassDistance()
        self.K += 1
        flag = False
        while (self.K <= max_K) and (flag is False):
            self._init_centroids()
            self.fit()
            w = self.withinClassDistance()
            if 100 - (w / aux) * 100 < 20:
                self.K -= 1
                flag = True
            else:
                self.K += 1
                aux = w
        self._init_centroids()
        self.fit()
        aux = self.withinClassDistance()


def distance(X, C):

    return np.sqrt((X[:, 0, np.newaxis] - C[:, 0]) ** 2 + (X[:, 1, np.newaxis] - C[:, 1]) ** 2 + (
                X[:, 2, np.newaxis] - C[:, 2]) ** 2)


def get_colors(centroids):

    color_probs = utils.get_color_prob(centroids)
    labels = np.empty(len(centroids), dtype=object)[:]
    labels[:] = np.nan

    for c in range(len(centroids)):
        labels[c] = utils.colors[np.argmax(color_probs[c])]

    return labels

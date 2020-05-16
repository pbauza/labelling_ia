__authors__ = ['1491858', '1493406', '1493962']
__group__ = 'DL.15'

import numpy as np
import utils
from math import floor
from utils_data import *


class KMeans:

    def __init__(self, X, K=1, options=None):

        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X):

        if len(X[0] != 3):
            X = X.reshape(len(X)*len(X[0]), 3)

        self.X = X.astype(float)

    def _init_options(self, options=None):

        if options is None:
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
            options['fitting'] = 'WCD'

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

        elif self.options['km_init'].lower() == 'random':
            for pixel in np.random.permutation(self.X):
                if not any(np.equal(pixel, self.centroids).all(1)):
                    self.centroids[index_pixel] = pixel
                    index_pixel += 1
                    if index_pixel == self.K:
                        break

        elif self.options['km_init'].lower() == 'custom':
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for element in self.X:
                sum1 += element[0]
                sum2 += element[1]
                sum3 += element[2]
            mean1 = sum1 / 4800
            mean2 = sum2 / 4800
            mean3 = sum3 / 4800

            for i in range(self.K):
                if i % 2 == 0:
                    self.centroids[i] = [mean1 + ((i * mean1) / self.K), mean2 + ((i * mean2) / self.K),
                                         mean3 + ((mean3 * i) / self.K)]
                else:
                    self.centroids[i] = [mean1 - ((i * mean1) / self.K), mean2 - ((i * mean2) / self.K),
                                         mean3 - ((i * mean3) / self.K)]

        pass


    def get_labels(self):

        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)


    def get_centroids(self):

        self.old_centroids = np.array(self.centroids, copy=True)
        m = np.bincount(self.labels, minlength=self.K)
        m = m.reshape(-1,1)
        for ind, n in enumerate(m):
            if n[0] == 0:
                m[ind] += 1
        self.centroids = [self.X[self.labels == index].sum(0) for index in range(
            0, len(self.centroids))] / m


    def converges(self):

        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        self._init_centroids()
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

    def interClassDistance(self):

        dist = 0
        for i in range(0, len(self.centroids)):
            pixels_per_centroids = np.where(self.labels == i)[0]
            # AGAFEM TOTS ELS CENTROIDES DIFERENTS AL CENTROIDE DE LA CLASSE RESPECTE A LA QUE MESUREM DISTANCIES
            m = self.centroids[np.where(np.array(range(0, len(self.centroids))) != i)]
            #PER CADA CENTROIDE QUE DIFERENT, CALCULEM LA DISTANCIA A CADA PIXEL DE LA CLASE QUE TOCA
            for c in m:
                #AQUESTA DIST LA SUMEM A UNA VARIABLE QUE DESPRÉS DIVIDIREM ENTRE EL NOMBRE DE PIXELS COM FEIEM A LA
                #WITHIN CLASS DISTANCE
                dist += np.sum((self.X[pixels_per_centroids[:]]-c)**2)
        return dist/len(self.X) #AQUESTA DIST VOLEM QUE SIGUI GRAN

    def fisherDiscriminant(self):
        '''Fisher's Discriminant: (d_intra class) / (d_inter class)'''

        return self.withinClassDistance() / self.interClassDistance()

    def find_bestK(self, max_K):

        self.K = 2
        self.fit()
        aux = self.withinClassDistance()
        self.K += 1
        flag = False
        while (self.K <= max_K) and (flag is False):
            self.fit()
            w = self.withinClassDistance()
            percent = (w / aux) * 100
            if 100 - percent < 20:
                self.K -= 1
                flag = True
            else:
                self.K += 1
                aux = w
        if flag is False:
            self.K = max_K
        self.fit()

    def find_bestKImprovement(self, max_K, value, type):

        self.K = 2
        self.fit()
        if type == 'Inter':
            aux = self.interClassDistance()
        elif type == 'Fisher':
            aux = self.fisherDiscriminant()
        else:
            aux = self.withinClassDistance()

        self.K += 1
        flag = False
        while (self.K <= max_K) and (flag is False):
            self.fit()
            if type == 'Inter':
                w = self.interClassDistance()
                percent = (aux / w) * 100
            elif type == 'Fisher':
                w = self.fisherDiscriminant()
                percent = (w / aux) * 100
            else:
                w = self.withinClassDistance()
                percent = (w / aux) * 100

            if 100 - percent < value:
                self.K -= 1
                flag = True
            else:
                self.K += 1
                aux = w
        if flag is False:
            self.K = max_K

        self.fit()

def distance(X, C):

    return np.sqrt((X[:, 0, np.newaxis] - C[:, 0]) ** 2 + (X[:, 1, np.newaxis] - C[:, 1]) ** 2 + (
                X[:, 2, np.newaxis] - C[:, 2]) ** 2)


def get_colors(centroids):

    return [utils.colors[np.argmax(utils.get_color_prob(centroids)[c])] for c in range(len(centroids))]

def printCustomPoints(centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for x, y, z in centroids:
        #xs = randrange(n, 23, 32)
        #ys = randrange(n, 0, 100)
        #zs = randrange(n, zlow, zhigh)
        ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

#NAIVE SHARDING CODE:

# def naive_sharding(ds, k):
#     """
#     Create cluster centroids using deterministic naive sharding algorithm.
#
#     Parameters
#     ----------
#     ds : numpy array
#         The dataset to be used for centroid initialization.
#     k : int
#         The desired number of clusters for which centroids are required.
#     Returns
#     -------
#     centroids : numpy array
#         Collection of k centroids as a numpy array.
#     """
#
#     #index = 0
#     n = np.shape(ds)[1]
#     m = np.shape(ds)[0]
#     centroids = np.mat(np.zeros((k, n)))
#     #centroids = np.empty([k, n], float)
#
#     # Sum all elements of each row, add as col to original dataset, sort
#     composite = np.mat(np.sum(ds, axis=1))
#     ds = np.append(composite.T, ds, axis=1)
#
#     #composite = np.sum(ds, axis=1)
#     #for elComp, elDs in zip(composite, ds):
#         #np.insert(elDs, 0, elComp)
#         #ds[index] = elDs
#         #index += 1
#
#
#     ds.sort(axis=0)
#
#     # Step value for dataset sharding
#     step = floor(m / k)
#
#
#     # Vectorize mean ufunc for numpy array
#     vfunc = np.vectorize(_get_mean)
#
#     # Divide matrix rows equally by k-1 (so that there are k matrix shards)
#     # Sum columns of shards, get means; these columnar means are centroids
#     for j in range(k):
#         if j == k - 1:
#             centroids[j:] = vfunc(np.sum(ds[j * step:, 1:], axis=0), step)
#         else:
#             centroids[j:] = vfunc(np.sum(ds[j * step:(j + 1) * step, 1:], axis=0), step)
#
#     return centroids
#
#
# def _get_mean(sums, step):
#     """
#     Vectorizable ufunc for getting means of summed shard columns.
#
#     Parameters
#     ----------
#     sums : float
#         The summed shard columns.
#     step : int
#         The number of instances per shard.
#     Returns
#     -------
#     sums/step (means) : numpy array
#         The means of the shard columns.
#     """
#
#     return sums / step
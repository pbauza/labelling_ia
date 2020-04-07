__authors__ = ['1491858','1493406', '1493962']
__group__ = 'DL15'

import numpy as np
import copy
import utils
import math

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """

        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self._init_centroids()
        self.index = np.zeros([self.K], int)

    def _init_X(self, X):

        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """

        if len(X[0] != 3):
            X = X.reshape(len(X)*len(X[0]), 3) #numpy(numpy(numpy(float,float,float))) X[19][0][1/2/3]

        self.X = X.astype(float)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
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

        """
        Initialization of centroids
        """
        self.centroids = np.zeros([self.K, 3], float)
        self.old_centroids = np.zeros([self.K, 3], float)
        aux = np.zeros([self.K, 3], float)
        index_pixel = 0
        if self.options['km_init'].lower() == 'first':
            for pixel in self.X:
                if not any(np.equal(pixel, aux).all(1)):
                    aux[index_pixel] = pixel
                    index_pixel += 1
                    if index_pixel == self.K:
                        self.centroids = aux
                        break
        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        #elif self.options['km_init'].lower() == 'custom':

    def get_labels(self):

        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        self.labels = np.empty(len(self.X), int)
        distances = distance(self.X, self.centroids)

        for j,d in enumerate(distances):
            min = d.min()
            i, = np.where(d == min)
            self.labels[j] = i[0]
            self.index[i[0]] += 1

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = copy.copy(self.centroids[:])

        aux = np.empty([self.K], np.object)

        for index_pixel, index_centroid in enumerate(self.labels):
            if aux[index_centroid] is None:
                aux[index_centroid] = []
            aux[index_centroid].append(self.X[index_pixel])

        for index_centroid, points in enumerate(aux):
            # MANERA COMPACTA:
            self.centroids[index_centroid] = np.array([sum(i) / len(points) for i in zip(*points)])

        # # Copiem el self.centroids al self.old_centroids tal i com ens diu l'enunciat
        # self.old_centroids = copy.copy(self.centroids[:])
        #
        # aux = np.zeros([self.K, self.index.max(), 3], float)
        # index = np.zeros([self.K], int)
        #
        # for index_pixel, index_centroid in enumerate(self.labels):
        #     aux[index_centroid][index[index_centroid]] = self.X[index_pixel]
        #     index[index_centroid] += 1
        # self.centroids = np.transpose(aux.T.sum(1)/(self.index[:]))

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.equal(self.centroids, self.old_centroids).all()

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        while (self.converges() != True):
            self.get_labels()
            self.get_centroids()
            self.num_iter = self.num_iter + 1

    def withinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        aux = np.empty([self.K], np.object)
        wcd = np.zeros([self.K], float)
        dist = 0

        '''
        for index_pixel, index_centroid in enumerate(self.labels):
            if aux[index_centroid] is None:
                aux[index_centroid] = []
            aux[index_centroid].append(self.X[index_pixel])


        for i, cx in enumerate(aux):
            x = np.zeros([1, 3], float)
            x[0] = cx[i]
            dist_array = distance(x, np.array(cx))
            #for y in cx:
            #for y in dist_array:
                #dist += math.sqrt(pow(x[0]-y[0], 2) + pow(x[1]-y[1], 2) + pow(x[2]-y[2], 2))
                #dist += y
            dist = np.sum(dist_array)
            self.wcd[i] = dist/len(cx)
        '''

        for index_pixel, index_centroid in enumerate(self.labels):
            if aux[index_centroid] is None:
                aux[index_centroid] = []
            aux[index_centroid].append(self.X[index_pixel])


        for i, aux_row in enumerate(aux):
            for pixel in aux_row:
                # AQUEST WHILE ES POT OPTIMITZAR. SI HO INTERPRETÈSSIM COM UNA MATRIU, SERIA SIMÈTRICA I PER TANT, PODRÍEM FER LA MEITAT D'OPERACIONS
                counter = 0
                while counter < len(aux_row):
                    dist += math.sqrt(pow(pixel[0] - aux_row[counter][0], 2) + pow(pixel[1] - aux_row[counter][1], 2) + pow(pixel[2] - aux_row[counter][2], 2))
                    counter += 1
            wcd[i] = dist/len(aux_row)
            dist = 0

        return np.sum(wcd)/self.K

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        self.K = 2
        self._init_centroids()

        self.fit()
        aux = self.withinClassDistance()
        self.K += 1
        flag = False

        while (self.K <= max_K) and (flag != True):
            self._init_centroids()
            self.fit()
            #wcd = np.zeros([self.K], float)
            wcd = self.withinClassDistance()
            newDec = (wcd/aux)*100
            print(newDec)
            if((100-newDec) < 20):
                flag = True
            else:
                self.K += 1
                aux = newDec


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    return np.sqrt(((X[:, :, None] - C[:, :, None].T) ** 2).sum(1))


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)

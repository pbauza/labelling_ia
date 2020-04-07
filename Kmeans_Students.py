__authors__ = ['1491858','1493406', '1493962']
__group__ = 'DL15'

import numpy as np
import copy
import utils

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

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################


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

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        '''        
        self.centroids = np.empty([self.K, 3], float)
        self.old_centroids = np.empty([self.K, 3], float)
        index_pixel = 1
        found = False
        if self.options['km_init'].lower() == 'first':
            self.centroids[0] = self.X[0]
            for pixel in self.X[1:]:
                found = False
                index_centroids = 0
                while index_centroids < index_pixel and found is False:
                    if (pixel == self.centroids[index_centroids]).all():
                        found = True
                    index_centroids += 1;
                if found is False:
                    self.centroids[index_pixel] = pixel
                    self.old_centroids[index_pixel] = pixel
                    index_pixel += 1
                    if index_pixel == self.K:
                        break
        '''

        self.centroids = np.zeros([self.K, 3], float)
        self.centroids = np.zeros([self.K, 3], float)
        aux = np.zeros([self.K, 3], float)
        index_pixel = 0
        if self.options['km_init'].lower() == 'first':
            for pixel in self.X:
                if not any(np.equal(pixel, aux).all(1)):
                    aux[index_pixel] = pixel
                    index_pixel += 1
                    if index_pixel == self.K:
                        self.centroids = aux
                        self.old_centroids = aux
                        break


        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        #elif self.options['km_init'].lower() == 'custom':


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.labels = np.random.randint(self.K, size=self.X.shape[0])

        self.labels = np.empty(len(self.X), int)
        distances = distance(self.X, self.centroids)

        for j,d in enumerate(distances):
            min = d.min()
            i, = np.where(d == min)
            self.labels[j] = i[0]

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        # Copiem el self.centroids al self.old_centroids tal i com ens diu l'enunciat
        self.old_centroids = copy.copy(self.centroids[:])

        aux = np.empty([self.K], np.object)

        for index_pixel, index_centroid in enumerate(self.labels):
            if aux[index_centroid] is None:
                aux[index_centroid] = []
            aux[index_centroid].append(self.X[index_pixel])

        for index_centroid, points in enumerate(aux):
            # MANERA COMPACTA:
            self.centroids[index_centroid] = np.array([sum(i)/len(points) for i in zip(*points)])
            # MANERA NO COMPACTA:
            # sumx = 0
            # sumy = 0
            # sumz = 0
            # for pixel in points:
            #     long = len(points)
            #     sumx += pixel[0]
            #     sumy += pixel[1]
            #     sumz += pixel[2]
            # self.centroids[index_centroid] = (sumx / long, sumy / long, sumz / long)


    '''
        aux = np.empty(self.K, float)
        aux_centroid = np.empty(2, float)

        # Classifiquem els pixels segons el centroid al qual estiguin associats (contingut del self.labels)
        for element, pixel in zip(self.labels, self.X):
            if aux[element] is None:
                aux[element] = pixel
            else:
                aux[element] = np.append(aux[element], pixel)

        # Calculem el nou centroid per a cadascun dels groups
        for group, index in enumerate(aux):
            length = group.shape[0]
            sum_x = np.sum(group[:, 0])
            sum_y = np.sum(group[:, 1])
            sum_z = np-sum(group[:, 2])
            self.centroids[index] = copy.deepcopy(np.array(np.array((sum_x/length), (sum_y/length), (sum_z/length)))

        #LINK algorisme: https://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi#23021198
    '''
    def converges(self):

        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        value = np.mean(self.centroids - self.old_centroids)
        if value == 0:
            res = True
        else:
            res = False
        return res

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.rand()

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


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

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    distaux = []
    for filaI in X:
        var = []
        for filaC in C:
            var.append(np.linalg.norm(filaI - filaC))
        distaux.append(var)
    dist = np.array(distaux)
    return dist


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

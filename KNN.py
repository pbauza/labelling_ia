__authors__ = ['1491858', '1493406', '1493962']
__group__ = 'DL.15'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from scipy.stats import mode

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        #self.train_data es una matriu de P (número d'imatges) x 4800
        self.train_data = train_data.reshape(len(train_data), len(train_data[0]) * len(train_data[0][1])).astype(float)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        # test_data = test_data.reshape(len(test_data), len(test_data[0]) * len(test_data[0][1])).astype(float)
        # self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data))[:, :k][:]]
        self.neighbors = self.labels[np.argsort(cdist(test_data.reshape(len(test_data), len(test_data[0]) * len(
            test_data[0][1])).astype(float), self.train_data))[:, :k][:]]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        mostVotedValues = np.array([], dtype='<U8')
        #clothes = np.array([], dtype='<U8')
        percent = np.empty([len(self.train_data), 1])
        aux_dict = dict()
        clothes = []

        #bàsicament he de recorrer el neighbours, agafar el primer element i anar-lo posant a la casella corresponent
        for i, element in enumerate(self.neighbors):
            for label in element:
                if label not in aux_dict.keys():
                    aux_dict[label] = 0
                aux_dict[label] += 1

            highest = max(aux_dict.values())

            [clothes.append(el) for el in aux_dict.keys() if aux_dict[el] == highest]

            mostVotedValues = np.append(mostVotedValues, clothes[0])
            percent[i] = highest/sum(aux_dict.values())
            clothes = []
            aux_dict = dict()

        # RESULT: y: array(['Handbags', 'Dresses', 'Jeans', 'Dresses', 'Heels', 'Heels', 'Dresses', 'Shorts', 'Dresses', 'Heels'], dtype='<U8')

        return mostVotedValues#, percent


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at

        !!!!! :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got !!! --> Diu que retorna dos arrays, pero al fitxer de test ho té
        com si únicament li retornessin un paràmetre (no sé si ho he de retornar com un zip o un únic array o klk.
        """

        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

        self.get_k_neighbours(test_data, k)
        mostVotedValues = self.get_class()

        return mostVotedValues


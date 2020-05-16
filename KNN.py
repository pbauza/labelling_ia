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

    def _init_train(self,train_data):
        """
        Initializes the train data
        + Args:
            - train_data: PxMxNx3 matrix corresponding to P color images (number_images) x 4800
        + Return:
            - assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)

        """

        self.train_data = train_data.reshape(len(train_data), train_data[0].size).astype(float)


    def get_k_neighbours(self, test_data, k):
        """
        Given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        + Args:
            - test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
            - k:  the number of neighbors to look at
        + Return:
            - the matrix self.neighbors is created (NxK) the ij-th entry is the j-th nearest train point to the i-th test point

        """

        test_data = test_data.reshape(len(test_data), test_data[0].size).astype(float)
        self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data))[:, :k][:]] #Agafem labels dels K primers veïns (de cada test data) amb la cdist més curta


    def get_class(self):
        """
        Get the class by maximum voting
        + Return:
            - 2 numpy array of Nx1 elements.
                - 1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                - 2nd array For each of the rows in self.neighbors gets the % of votes for the winning class

        """

        mostVotedValues = np.array([], dtype='<U8')
        percent = np.empty([len(self.neighbors), 1])
        aux_dict = dict()
        clothes = []

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

        return mostVotedValues


    def predict(self, test_data, k):
        """
        Predicts the class at which each element in test_data belongs to
        + Args:
            - test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
            - k: the number of neighbors to look at

        Return:
             + the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got

        (!!!) Note: Although it says it returns two parameters, the test file only accepts one

        """

        self.get_k_neighbours(test_data, k)
        mostVotedValues = self.get_class()

        return mostVotedValues


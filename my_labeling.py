__authors__ = ['1491858', '1493406', '1493962']
__group__ = 'DL.15'

import numpy as np
import Kmeans_Students
import KNN
from KNN import *
from Kmeans_Students import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
#import cv2
from operator import  itemgetter
import random

## You can start coding your functions here

#ANALISIS QUALITATIU

def retrieval_by_color(images, labels, colors):
    c = np.empty([len(labels)], np.object)
    c[:] = np.nan
    p = np.empty([len(labels)], np.object)
    p[:] = np.nan

    for i in range(len(labels)):
        c[i], p[i] = np.unique(labels[i], return_counts=True)
        p[i] = p[i] / len(labels[i])

    image_list = dict()
    for index, i in enumerate(images):
        m = 0
        found = False
        while m < len(colors) and found is False:
            if colors[m] not in labels[index]:
                found = True
                if index in image_list.keys():
                    image_list.pop(index, None)
            else:
                if index not in image_list.keys():
                    image_list[index] = 0
                image_list[index] += p[index][np.argwhere(c[index] == colors[m])]
                m += 1

    image_list = sorted(image_list.items(), key=itemgetter(1), reverse=True)
    return_list = list()

    for i in range(len(image_list)):
        return_list.append(images[image_list[i][0]])

    visualize_retrieval(return_list, 4)

def retrieval_by_shape(images, labels, shapes):
    c = np.empty([len(labels)], np.object)
    c[:] = np.nan
    p = np.empty([len(labels)], np.object)
    p[:] = np.nan

    for i in range(len(labels)):
        c[i], p[i] = np.unique(labels[i], return_counts=True)
        p[i] = p[i] / len(labels[i])

    image_list = dict()
    for index, i in enumerate(images):
        m = 0
        found = False
        while m < len(shapes) and found is False:
            if shapes[m] not in labels[index]:
                found = True
                if index in image_list.keys():
                    image_list.pop(index, None)
            else:
                if index not in image_list.keys():
                    image_list[index] = 0
                image_list[index] += p[index][np.argwhere(c[index] == shapes[m])]
                m += 1

    image_list = sorted(image_list.items(), key=itemgetter(1), reverse=True)
    return_list = list()

    for i in range(len(image_list)):
        return_list.append(images[image_list[i][0]])

    visualize_retrieval(return_list, 2) #no sabem per que no funciona si esta igual que laltre


#ANALISI QUANTITATIU

def kmean_statistics(kmeans, kmax):
    for i in range(2, kmax):
        kmeans.find_bestK(i)
        shape = kmeans.X.shape
        visualize_k_means(kmeans, shape)
        #return kmeans.num_iter

def get_shape_accuracy(labels, gt):
    print("SHAPE ACCURACY: ", 100*sum(1 for x, y in zip(sorted(labels), sorted(gt)) if x == y)/len(labels), "%")

def get_color_accuracy(labels, gt):
    c = np.empty([len(labels)], np.object)
    c[:] = np.nan
    p = np.empty([len(labels)], np.object)
    p[:] = np.nan

    for i in range(len(labels)):
        c[i], p[i] = np.unique(labels[i], return_counts=True)
        p[i] = p[i] / len(labels[i])

    labels = []
    for index, i in enumerate(p):
        m = np.argwhere(i == np.amax(i)).flatten().tolist()
        labels.append(c[index][m].tolist())


    print("COLOR ACCURACY: ", 100*sum(1 for x, y in zip(labels, gt) if x == y)/len(labels), "%")


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    #Test Qualitative Fuctions
    retrieval_by_color(test_imgs, test_color_labels, ['Yellow', 'Black'])
    retrieval_by_shape(test_imgs, test_class_labels, ['Socks'])

    #Test quantitative functions
    knn = KNN(train_imgs[0:750], test_class_labels[0:750])
    knn.predict(test_imgs[0:750], 4)
    get_shape_accuracy(knn.labels, train_class_labels[0:750])

    list = []
    for i in range(0, 20):
        km = KMeans(test_imgs[i], 2)
        km.find_bestK(10)
        list.append(get_colors(km.centroids))

    get_color_accuracy(list, test_color_labels[:20])






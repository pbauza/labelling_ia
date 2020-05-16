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

def retrieval_by_color(images, labels, colors, fig_name):
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
    return_list = []

    for i in range(len(image_list)):
        return_list.append(images[image_list[i][0]])

    visualize_retrieval(return_list, 4, fig_name=fig_name)

    return return_list

def retrieval_by_shape(images, labels, shapes, fig_name):
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
    return_list = []

    for i in range(len(image_list)):
        return_list.append(images[image_list[i][0]])

    visualize_retrieval(return_list, 4, fig_name=fig_name) #no sabem per que no funciona si esta igual que laltre

    return return_list

#ANALISI QUANTITATIU

def kmean_statistics(kmeans, kmax):
    for i in range(2, kmax):
        kmeans.find_bestKImprovement(i, 80, 'Intra')
        shape = kmeans.X.shape
        visualize_k_means(kmeans, shape, i)

def get_shape_accuracy(labels, gt):
    return str(sum(1 for x, y in zip(labels, gt) if x == y)/len(labels))

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

    suma = 0; aux = 0
    for x, y in zip(labels, gt):
        aux = 0
        for i in x:
            if i in y:
                aux += 1
        suma += aux/len(y)

    return str(suma / len(labels))


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # lista = []
    # for i in range(0, 750):
    #     km = KMeans(test_imgs[i], 2)
    #     km.find_bestKImprovement(10, 20, 'Intra')
    #     lista.append(get_colors(km.centroids))
    # retrieval_by_color(test_imgs[:750], lista, ['Pink'],
    #                    "./imatges_proves/defecte/" + "pink" + "_" + "meh" + ".png")

    # knn = KNN(train_imgs[:500], train_class_labels[:500])
    # preds = knn.predict(test_imgs[:150], 4)
    # retrieval_by_shape(test_imgs[:150], preds, ['Sandals'],
    #                    "./imatges_proves/shape/" + "sd" + "_" + "500" + ".png")


    #Test Qualitative Fuctions
    # retrieval_by_color(test_imgs, test_color_labels, ['Green', 'Blue'], "./imatges_proves/color/color1.png")
    # retrieval_by_shape(test_imgs, test_class_labels, ['Socks'], "./imatges_proves/shape/shape1.png")

    #Test quantitative functions

    # element_kmeans = KMeans(test_imgs[10])
    # kmean_statistics(element_kmeans, 10)

    n_images_c = 50
    n_images_s = 150
    f = open('proves_color_' + str(n_images_c) + 'img.txt', 'w')
    f1 = open('proves_shape_'+ str(n_images_s) + 'img.txt', 'w')
    f.write("Iteration, Values, Type" + "\n")
    f1.write("Iteration, Values, Type" + "\n")

    types = ['Inter', 'Intra', 'Fisher']

    list = []
    n = 20
    t = 'Intra'
    f.write(str(-1) + ',' + str(n) + "," + t + ",")
    for i in range(0, n_images_c):
        km = KMeans(test_imgs[i], 2)
        km.find_bestKImprovement(10, n, t)
        list.append(get_colors(km.centroids))
    f.write(get_color_accuracy(list, test_color_labels[:40]) + "\n")
    retrieval_by_color(test_imgs[:n_images_c], list, ['Red', 'White'],
                       "./imatges_proves/color/" + "rw" + "_" + str(n) + "_" + t + ".png")

    for it in range(0, 1):

        n = random.randrange(10, 90)

        t = types[0]
        f.write(str(it) + "," + str(n) + "," + t + ",")
        list = []
        for i in range(0, n_images_c):
            km = KMeans(test_imgs[i], 2)
            km.find_bestKImprovement(10, n, t)
            list.append(get_colors(km.centroids))
        f.write(get_color_accuracy(list, test_color_labels[:n_images_c]) + "\n")
        retrieval_by_color(test_imgs[:n_images_c], list, ['Red', 'White'],
                           "./imatges_proves/color/" + "rw" + "_" + str(n) + "_" + t + ".png")

        t = types[1]
        f.write(str(it) + "," + str(n) + "," + t + ",")
        list = []
        for i in range(0, n_images_c):
            km1 = KMeans(test_imgs[i], 2)
            km1.find_bestKImprovement(10, n, t)
            list.append(get_colors(km1.centroids))
        f.write(get_color_accuracy(list, test_color_labels[:n_images_c]) + "\n")
        retrieval_by_color(test_imgs[:n_images_c], list, ['Red', 'White'],
                           "./imatges_proves/color/" + "rw" + "_" + str(n) + "_" + t + ".png")

        t = types[2]
        f.write(str(it) + "," + str(n) + "," + t + ",")
        list = []
        for i in range(0, n_images_c):
            km2 = KMeans(test_imgs[i], 2)
            km2.find_bestKImprovement(10, n, t)
            list.append(get_colors(km2.centroids))
        f.write(get_color_accuracy(list, test_color_labels[:n_images_c]) + "\n")
        retrieval_by_color(test_imgs[:n_images_c], list, ['Red', 'White'],
                           "./imatges_proves/color/" + "rw" + "_" + str(n) + "_" + t + ".png")

    # for it in range(0, 125):
    #     ti = random.randrange(10, 100)
    #     knn = KNN(train_imgs[:ti], train_class_labels[:ti])
    #     preds = knn.predict(test_imgs[:n_images_s], 4)
    #     f1.write(str(it) + "," + str(ti) + ",")
    #     f1.write(get_shape_accuracy(preds, test_class_labels[:n_images_s]) + "\n")
    #     retrieval_by_shape(test_imgs[:n_images_s], preds, [
    #         "Dresses"], "./imatges_proves/shape/" + "dresses" + "_" + str(ti) + ".png")


    #retrieval_by_color(test_imgs[0:40], list, ['Yellow', 'Black'])

    # f.close()
    # f1.close()







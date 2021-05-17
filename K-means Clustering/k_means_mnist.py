# -*- coding: utf-8 -*-
#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from display_network import *
#%%
mndata = MNIST('mnist_datasets/')
mndata.load_testing()
X = mndata.test_images
X0 = np.array(X)[:1000,:]/256.0
X = X0

kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)

#%%
print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU
import open3d as o3d
import matplotlib.pyplot as mplot
import numpy as np
#from tensorflow import keras
#from tensorflow.keras import layers
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Conv3D, MaxPooling3D, AveragePooling2D
from keras.utils import np_utils
from keras.datasets import mnist

"""
#Reading in the STL file
mesh = []

#loading the model for extra training purposes
model = load_model("model.h5")
print("Model Loaded")
model.summary()

for file in os.listdir('STLs'):
    print(file)
    mesh = (o3d.io.read_triangle_mesh("STLs/"+file))

    xtrain = []
    ytrain = []
    #Creates a x and y for the neural network using the vertices and triangles
    xtrain.append(np.asarray(mesh.vertices).astype('float32'))
    ytrain.append(np.asarray(mesh.triangles).astype('float32'))

    train = np.asarray(xtrain)
    target = np.asarray(ytrain)

    print(train.shape)

    history = model.fit(train, target, epochs=1000, verbose=1)
    model.save("model.h5")
"""

#reading the STL
mesh = []
xtrain = []
ytrain = []

mesh.append(o3d.io.read_triangle_mesh('cube.stl'))
mesh.append(o3d.io.read_triangle_mesh('Pyramid.stl'))

#Creates a x and y for the neural network using the vertices and triangles
for i in range(0, len(mesh)):
    xtrain.append(np.asarray(mesh[i].vertices).astype('float32'))
    ytrain.append(np.asarray(mesh[i].triangle_normals).astype('float32'))

xtrain = np.asarray(xtrain)
ytrain =  np.asarray(ytrain)

for i in range (0, len(mesh)):
    xtrain[i] = np.transpose(xtrain[i])
    ytrain[i] = np.transpose(ytrain[i])

#Code used to create and test the first trials of the code
train = (xtrain)
target = (ytrain)
train = np.expand_dims(xtrain,-1)
target = np.expand_dims(ytrain,-1)
#print(train.shape)
#print(target.shape)
print(train)
print(target)
inshape = (None,None,1)


#Building the Neural Network for the first time
model = Sequential()
model.add(Conv2D(32,kernel_size=(3), activation='relu', kernel_initializer='he_uniform', input_shape=inshape, data_format='channels_last'))
model.add(MaxPooling2D(padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=(3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='Poisson', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['poisson'])
model.summary()

#training the data
history = model.fit(train, target, epochs=10, verbose=1)

"""
#test = model.evaluate(train,target,verbose=1)
#initial save of the model
model.save("model.h5")
print("Model saved")


#This creates a point cloud and not a array of vertices. I may use this later.
#cloud = o3d.geometry.PointCloud()
#print(mesh[0])
#cloud.points = mesh[0].vertices
#o3d.visualization.draw_geometries([mesh[0].triangles])
"""
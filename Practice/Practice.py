import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import open3d as o3d


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]

def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    #print(np.asarray(data_t).shape)
    return np.asarray(data_t, dtype=np.float32)

with h5py.File("./full_dataset_vectors.h5", "r") as hf:
    xtrain = hf["X_train"][:]
    targets_train = hf["y_train"][:]
    xtest = hf["X_test"][:] 
    targets_test = hf["y_test"][:]

    sample_shape = (16,16,16,3)
    xtrain = rgb_data_transform(xtrain)
    xtest = rgb_data_transform(xtest)
    targets_train = to_categorical(targets_train).astype(np.integer)
    targets_test = to_categorical(targets_test).astype(np.integer)

print(xtrain)


model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

# Fit data to model
history = model.fit(xtrain, targets_train,
            batch_size=128,
            epochs=3,
            verbose=1,
            validation_split=0.3)

score = model.evaluate(xtest,targets_test,verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

model.save("model.h5")
print("Model saved")
"""

model = load_model("model.h5")
print("Model Loaded")
model.summary()
test = model.evaluate(xtest,targets_test,verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], test[1]*100))

a = []
a = np.zeros((2,3,4))
print(a)


mesh = (o3d.io.read_triangle_mesh("cube.stl"))
vert = np.asarray(mesh.vertices)
print(vert.ndim)



model.save("model.h5")
print("Model saved")

for i in range(1,len(mesh)):
    history = model.fit(train[i],target[i], epochs=1000, verbose=1)
model.save(model.h5)
print("Model saved")

"""

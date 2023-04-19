#!/usr/bin/env python
# coding: utf-8

# ### <font color='orange'><h3 align="center">Group - 1</h3></font>

# ### <font color='light blue'><h2 align="center">Traffic sign classification</h2></font>
# ### <font color='red'><h3 align="center">Experiments</h3></font>

# ### Importing Libraries

# In[3]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('C:/Users/arunt/OneDrive/Desktop/S6/AI/Mini Project')
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization 


# ### Storing data & labels

# In[4]:


data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()


# ### Load Data & Labels

# In[6]:


# Loading data and labels saved after data preprocessing in the main model

data=np.load('./trained_jupyterlab/data.npy')
labels=np.load('./trained_jupyterlab/target.npy')


# In[7]:


print(data.shape, labels.shape)


# ### Train - Test Split

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)


# In[9]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Convert labels to onehot encoding

# In[10]:


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


# ### Model - 1 : Change in number of filters

# In[11]:


model1 = Sequential()
model1.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
model1.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(BatchNormalization(axis=-1))
          
model1.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu'))
model1.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(BatchNormalization(axis=-1))
          
model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(rate=0.5))
          
model1.add(Dense(43, activation='softmax'))

#Compilation of the model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
fitting_model1 = model1.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[12]:


# Accuracy 
plt.figure(0)
plt.plot(fitting_model1.history['accuracy'], label='Training accuracy')
plt.plot(fitting_model1.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy - filters changed to 8,16; 8,16' )
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(fitting_model1.history['loss'], label='Training loss')
plt.plot(fitting_model1.history['val_loss'], label='Validation loss')
plt.title('Loss - filters changed to 8,16; 8,16')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# ### Model - 2 : Ignoring the dropout layers

# In[ ]:


model2 = Sequential()
model2.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
model2.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(BatchNormalization(axis=-1))
          
model2.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model2.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(BatchNormalization(axis=-1))
          
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(BatchNormalization())
          
model2.add(Dense(43, activation='softmax'))

#Compilation of the model
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
fitting_model2 = model2.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[ ]:


# Accuracy 
plt.figure(0)
plt.plot(fitting_model2.history['accuracy'], label='Training accuracy')
plt.plot(fitting_model2.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy - Ignoring dropout')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(fitting_model2.history['loss'], label='Training loss')
plt.plot(fitting_model2.history['val_loss'], label='Validation loss')
plt.title('Loss - Ignoring dropout')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# ### Model - 3 : Replacing batch normalization with dropout rates 0.25, 0.25, 0.5

# In[14]:


model3 = Sequential()
model3.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
model3.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model3.add(MaxPool2D(pool_size=(2, 2)))
model3.add(Dropout(rate=0.25))
          
model3.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model3.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model3.add(MaxPool2D(pool_size=(2, 2)))
model3.add(Dropout(rate=0.25))
          
model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(rate=0.5))
          
model3.add(Dense(43, activation='softmax'))

#Compilation of the model
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
fitting_model3 = model3.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[15]:


# Accuracy 
plt.figure(0)
plt.plot(fitting_model3.history['accuracy'], label='Training accuracy')
plt.plot(fitting_model3.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy - Droupout rates 0.25,0.25,0.5')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(fitting_model3.history['loss'], label='Training loss')
plt.plot(fitting_model3.history['val_loss'], label='Validation loss')
plt.title('Loss - Droupout rates 0.25,0.25,0.5')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# ### Model - 4 : Altering dropout rates to 0.1, 0.1, 0.1

# In[16]:


model4 = Sequential()
model4.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
model4.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model4.add(MaxPool2D(pool_size=(2, 2)))
model4.add(Dropout(rate=0.1))
          
model4.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model4.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model4.add(MaxPool2D(pool_size=(2, 2)))
model4.add(Dropout(rate=0.1))
          
model4.add(Flatten())
model4.add(Dense(512, activation='relu'))
model4.add(Dropout(rate=0.1))
          
model4.add(Dense(43, activation='softmax'))

#Compilation of the model
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
fitting_model4 = model4.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[17]:


# Accuracy 
plt.figure(0)
plt.plot(fitting_model4.history['accuracy'], label='Training accuracy')
plt.plot(fitting_model4.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy - Droupout rates 0.1,0.1,0.1')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(fitting_model4.history['loss'], label='Training loss')
plt.plot(fitting_model4.history['val_loss'], label='Validation loss')
plt.title('Loss - Droupout rates 0.1,0.1,0.1')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# ### Model - 5 : Changing Optimizers to "rmsprop" and "sgd"

# In[18]:


model5 = Sequential()
model5.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
model5.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model5.add(MaxPool2D(pool_size=(2, 2)))
model5.add(BatchNormalization(axis=-1))
          
model5.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model5.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model5.add(MaxPool2D(pool_size=(2, 2)))
model5.add(BatchNormalization(axis=-1))
          
model5.add(Flatten())
model5.add(Dense(512, activation='relu'))
model5.add(BatchNormalization())
model5.add(Dropout(rate=0.5))
          
model5.add(Dense(43, activation='softmax'))

#Compilation of the model
model5.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

epochs = 20
fitting_model5 = model5.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[19]:


#Compilation of the model
model5.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

epochs = 20
fitting_model6 = model5.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[21]:


# Training accuracy 
plt.figure(0)
plt.plot(fitting_model5.history['accuracy'], label='rmsprop')
plt.plot(fitting_model6.history['accuracy'], label='sgd')
plt.title('Training Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Training loss
plt.plot(fitting_model5.history['loss'], label='rmsprop')
plt.plot(fitting_model6.history['loss'], label='sgd')
plt.title('Training loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Validation accuracy 
plt.figure(0)
plt.plot(fitting_model5.history['val_accuracy'], label='rmsprop')
plt.plot(fitting_model6.history['val_accuracy'], label='sgd')
plt.title('Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Validation loss
plt.plot(fitting_model5.history['val_loss'], label='rmsprop')
plt.plot(fitting_model6.history['val_loss'], label='sgd')
plt.title('Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
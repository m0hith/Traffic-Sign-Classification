#!/usr/bin/env python
# coding: utf-8

# ### <font color='orange'><h3 align="center">Group - 1</h3></font>

# ### <font color='light blue'><h2 align="center">Traffic sign classification</h2></font>

# ### Importing Libraries

# In[1]:


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

# In[2]:


data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()


# ### Image Pre-Processing

# In[4]:


for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)


# ### Converting lists to numpy arrays

# In[5]:


data = np.array(data)
labels = np.array(labels)


# ### Save Labels & Data for future use

# In[6]:


# os.mkdir('trained_jupyterlab')
np.save('./trained_jupyterlab/data',data)
np.save('./trained_jupyterlab/target',labels)


# ### Load Data & Labels

# In[7]:


data=np.load('./trained_jupyterlab/data.npy')
labels=np.load('./trained_jupyterlab/target.npy')


# In[8]:


print(data.shape, labels.shape)


# ### Train - Test Split

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)


# In[10]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Convert labels to onehot encoding

# In[11]:


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


# ### Model building

# In[12]:


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
          
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
          
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
          
model.add(Dense(43, activation='softmax'))


# In[13]:


#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### Training

# In[14]:


epochs = 20
fitting_model = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[26]:


# accuracy 
plt.figure(0)
plt.plot(fitting_model.history['accuracy'], label='training accuracy')
plt.plot(fitting_model.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[27]:


# Loss
plt.plot(fitting_model.history['loss'], label='training loss')
plt.plot(fitting_model.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# ### Testing

# In[28]:


def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test,label


# In[29]:


X_test, label = testing('Test.csv')


# In[30]:


Y_pred = model.predict_classes(X_test)
Y_pred


# ### Accuracy with the test data

# In[31]:


from sklearn.metrics import accuracy_score
print(accuracy_score(label, Y_pred))


# ### Save the model

# In[32]:


model.save("./trained_jupyterlab/TSR.h5")


# ### Load the model

# In[33]:


import os
os.chdir(r'C:\Users\arunt\OneDrive\Desktop\S6\AI\Mini Project')
from keras.models import load_model
model = load_model('./trained_jupyterlab/TSR.h5')


# In[34]:


# Classes of trafic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


# In[35]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return image,Y_pred


# In[36]:


plot,prediction = test_on_img(r'C:\Users\arunt\OneDrive\Desktop\pic.jpg')
s = [str(i) for i in prediction] 
a = int("".join(s)) 
print("Predicted traffic sign is: ", classes[a])
plt.imshow(plot)
plt.show()


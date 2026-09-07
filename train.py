import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/validation'

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for image_name in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir,label,image_name))
            labels.append(label)
        print(label, "Completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)
print(train)


test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)
print(test)


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale", target_size=(48,48))
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48,1)
    return features


train_features = extract_features(train['image'])
test_features = extract_features(test['image'])

X_train = train_features/255.0
X_test = test_features/255.0

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

encoder = LabelEncoder()
encoder.fit(train['label'])

y_train = encoder.transform(train['label'])
y_test =  encoder.transform(test['label'])

y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, batch_size=32, epochs=14, validation_data=(X_test, y_test))

model_json = model.to_json()
with open("model/emotiondetector.json","w") as json_file:
    json_file.write(model_json)
model.save("model/emotiondetector.h5")
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Dense, Flatten, Lambda
from keras.layers import Convolution2D, Dropout
from keras.layers.pooling import MaxPooling2D 
from keras import backend

def generator(samples, batch_size=32):
    num_samples = len(samples)
    ANGLE_ADJUSTMENT = 0.15
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name) #BGR
                # BGR to RGB since drive.py reads images in RGB
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # filp images left right
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                
                # left image
                name = './IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                # BGR to RGB since drive.py reads images in RGB
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + ANGLE_ADJUSTMENT
                images.append(left_image)
                angles.append(left_angle)
                # filp images left right
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)
                
                # right image
                name = './IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                # BGR to RGB since drive.py reads images in RGB
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3]) - ANGLE_ADJUSTMENT
                images.append(right_image)
                angles.append(right_angle)
                # filp images left right
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        angle = float(line[3])
        if angle == 0 and np.random.random_sample() > 0.5:
                continue
        samples.append(line)
print (len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 90, 320  # Trimmed image format

backend.set_image_data_format('channels_last')
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Convolution2D(48,3,3, subsample = (2,2), activation = 'relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Convolution2D(64,3,3, subsample = (2,2), activation = 'relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/5, \
                    validation_data=validation_generator, validation_steps=len(validation_samples)/5, \
                    epochs=5, verbose = 1)

#model.summary()
model.save('model2.h5')
print ('model saved.')
### plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

plt.savefig('loss2.jpg')
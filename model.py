import csv
import cv2
import math
import numpy as np
import pdb
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D

def read_log(filepath):
    lines = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def generate_data(data, batch_size):
    offset = 0.2
    correction = [0, offset, -offset]  # center, left, right cameras
    while True:
        cnt = 0
        imgs, angles = [], []
        for line in data:
            cnt += 1
            for i in range(3):
                img_path = line[i]
                img = cv2.imread(img_path)
                imgs.append(img)
                angle = float(line[3])
                angles.append(angle + correction[i])
                # flip images
                imgs.append(cv2.flip(img,1))
                angles.append(-1.0 * (angle + correction[i]))
            if cnt == batch_size:
                cnt = 0
                yield(np.array(imgs), np.array(angles));
                imgs, angles = [], []


# Read original data
lines_twolaps = read_log('/home/gaobiao/Documents/DeepLearning/zhihu_material/car_end2end/data/driving_log.csv')
lines_curves = read_log('/home/gaobiao/Documents/DeepLearning/zhihu_material/car_end2end/data_curve/driving_log.csv')
lines_val = read_log('/home/gaobiao/Documents/DeepLearning/zhihu_material/car_end2end/data_val/driving_log.csv')
lines = np.array(lines_twolaps + lines_curves + lines_val)
# lines = np.array(lines_twolaps)

# Balance data
nbins = 2000
max_examples = 1000
balanced = np.empty([0, lines.shape[1]], dtype=lines.dtype)
for i in range(0, nbins):
    begin = i * (1.0 / nbins)
    end = begin + 1.0 / nbins
    extracted = lines[(abs(lines[:,3].astype(float)) >= begin) & (abs(lines[:,3].astype(float)) < end)]
    np.random.shuffle(extracted)
    extracted = extracted[0:max_examples, :]
    balanced = np.concatenate((balanced, extracted), axis=0)

np.random.shuffle(balanced)
trainData = balanced[0:int(balanced.shape[0] * 0.8), :]
valData = balanced[int(balanced.shape[0] * 0.8) + 1:, :]
valData = np.concatenate((valData, np.array(lines_val)), axis=0)
np.random.shuffle(valData)

print("TrainData : ", trainData.shape[0], " images.")
print("ValData : ", valData.shape[0], " images.")

# Build the model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001))
best_model = ModelCheckpoint('/home/gaobiao/Documents/DeepLearning/zhihu_material/car_end2end/model_best.h5', verbose=2, save_best_only=True)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=30, callbacks=[best_model])
model.fit_generator(generate_data(trainData, batch_size=64),
                    steps_per_epoch=math.ceil(trainData.shape[0] / 64),
                    epochs=200,
                    validation_data=generate_data(valData, batch_size=64),
                    validation_steps=math.ceil(valData.shape[0] / 64),
                    callbacks=[best_model])

# model.save('model_last.h5')

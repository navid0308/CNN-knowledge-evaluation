from keras.models import *
from keras.layers import *
from keras.optimizers import *
from load_data import *
from keras.callbacks import *

#import os.path

#while not os.path.exists('data/train.pklz'):
#    time.sleep(1)
train = pickle.load(gzip.open('data/train.pklz', 'rb'))
val = pickle.load(gzip.open('data/val.pklz', 'rb'))
test = pickle.load(gzip.open('data/test.pklz', 'rb'))

model = Sequential()

model.add(Conv2D(40, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=(img_row, img_col, 1)))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.50))

model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(Conv2D(60, (3, 3), activation='relu'))
model.add(Conv2D(70, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.20))

model.add(Conv2D(70, (2, 2), activation='relu'))
model.add(Conv2D(80, (2, 2), activation='relu'))
model.add(Conv2D(90, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005),
              metrics=['accuracy'])

filepath = 'weights-epoch-{epoch:02d}-val-loss-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
lr_adjust = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

model.fit(train.data, train.label,
            batch_size=32,
            epochs=100,
            verbose=1,
            validation_data=(val.data, val.label),
            callbacks=[checkpoint, lr_adjust])
from keras.models import load_model
model = load_model('weights-epoch-36-val-loss-0.27.hdf5')
score = model.evaluate(test.data, test.label, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
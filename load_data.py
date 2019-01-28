import os
import keras
import numpy as np
import cv2
import pickle
import gzip
import time

img_row = 64
img_col = 64

class ImageDataStore:
    def __init__(self, dataDir = None, train_folder = None, labels = None):
        self.data = []
        self.label = []

        if dataDir is not None and train_folder is not None and labels is not None:
            for label in labels:
                for root, _, filenames in os.walk(os.path.join(dataDir, train_folder, label)):
                    for file, i in zip(filenames, range(20000)):
                        self.Append(os.path.join(root, file), list(labels).index(label))
    def Append(self, path2img, label):
        self.data.append(cv2.resize(cv2.imread(path2img, cv2.IMREAD_GRAYSCALE), (0, 0), fx=0.5, fy=0.5))
        self.label.append(label)
    def Prep4Keras(self, row, col, channel):
        self.data = np.array(self.data).reshape(len(self.data), row, col, channel)
        self.label = keras.utils.to_categorical(self.label, len(list(set(self.label))))
    def Split(self, ratio):
        split1 = ImageDataStore()
        split2 = ImageDataStore()
        # assumes equal number of images per class
        img_per_class = int(len(self.label)/17)
        cutoff = int(img_per_class * ratio)
        for i in range(len(set(self.label))):
            curr_pos = img_per_class * i
            split1.data.extend(self.data[curr_pos : curr_pos + cutoff])
            split1.label.extend(self.label[curr_pos: curr_pos + cutoff])
            split2.data.extend(self.data[curr_pos + cutoff : curr_pos + img_per_class])
            split2.label.extend(self.label[curr_pos + cutoff: curr_pos + img_per_class])
        return split1, split2


if __name__ == "__main__":
    print('Preparing data for Keras...')
    start = time.time()
    dataDir = 'data/wallpapers'
    checkpointDir = 'modelCheckpoints'
    Symmetry_Groups = {'P1', 'P2', 'PM', 'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',
                        'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'}
    train_folder = 'train_aug'
    test_folder = 'test_aug'

    train = ImageDataStore(dataDir, train_folder, Symmetry_Groups)
    test = ImageDataStore(dataDir, test_folder, Symmetry_Groups)
    test, val = test.Split(0.5)

    train.Prep4Keras(img_row, img_col, 1)
    val.Prep4Keras(img_row, img_col, 1)
    test.Prep4Keras(img_row, img_col, 1)

    pickle.dump(train, gzip.open('data/train.pklz', 'wb'))
    pickle.dump(val, gzip.open('data/val.pklz', 'wb'))
    pickle.dump(test, gzip.open('data/test.pklz', 'wb'))
    end = time.time()
    print(str(end - start) + ' seconds taken to prep data.')
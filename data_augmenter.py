from augment import *
import os
import time

dataDir = 'data/wallpapers'
checkpointDir = 'modelCheckpoints'
Symmetry_Groups = {'P1', 'P2', 'PM', 'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',
                   'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'}

train_folder = 'train'
test_folder = 'test'

train_aug_folder = 'train_aug'
test_aug_folder = 'test_aug'

print('Augmenting data...')

if not os.path.exists(os.path.join(dataDir,train_aug_folder)):
    os.mkdir(os.path.join(dataDir,train_aug_folder))

for label in Symmetry_Groups:
    if not os.path.exists(os.path.join(dataDir, train_aug_folder,label)):
        os.mkdir(os.path.join(dataDir, train_aug_folder,label))

if not os.path.exists(os.path.join(dataDir,test_aug_folder)):
    os.mkdir(os.path.join(dataDir,test_aug_folder))

for label in Symmetry_Groups:
    if not os.path.exists(os.path.join(dataDir, test_aug_folder,label)):
        os.mkdir(os.path.join(dataDir, test_aug_folder,label))

start = time.time()
for label in Symmetry_Groups:
    for root, _, filenames in os.walk(os.path.join(dataDir, train_folder, label)):
        for file in filenames:
            for i in range(20):
                aug_img = augmentImage(cv2.imread(os.path.join(root,file)), scale=64)
                cv2.imwrite(os.path.join(dataDir, train_aug_folder, label, file.replace('.png', '_' + str(i+1) + '.png')), aug_img)
end = time.time()
print(str(end - start) + ' seconds taken to augment training data.')

start = time.time()
for label in Symmetry_Groups:
    for root, _, filenames in os.walk(os.path.join(dataDir, test_folder, label)):
        for file in filenames:
            for i in range(20):
                aug_img = augmentImage(cv2.imread(os.path.join(root, file)), scale=64)
                cv2.imwrite(os.path.join(dataDir, test_aug_folder, label, file.replace('.png', '_' + str(i+1) + '.png')), aug_img)
end = time.time()
print(str(end - start) + ' seconds taken to augment testing data.')
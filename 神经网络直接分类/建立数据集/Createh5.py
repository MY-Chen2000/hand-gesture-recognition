from scipy.misc import imread
import numpy as np
import pandas as pd
import os
import h5py
root = './gestures_test' # or ‘./test’ depending on for which the CSV is being created
f = h5py.File('test_myhand.h5','w')


def create():
    # go through each directory in the root folder given above
    for directory, subdirectories, files in os.walk(root):
    # go through each file in that directory
        for file in files:
        # read the image file and extract its pixels
            print(file)
            im = imread(os.path.join(directory,file))
            value = im.flatten()
    # I renamed the folders containing digits to the contained digit itself. For example, digit_0 folder was renamed to 0.
    # so taking the 9th value of the folder gave the digit (i.e. "./train/8" ==> 9th value is 8), which was inserted into the first column of the dataset.
            value = np.hstack((directory[11:],value))
            df = pd.DataFrame(value).T
            df = df.sample(frac=1) # shuffle the dataset
            with open('train_foo.csv', 'a') as dataset:
                df.to_csv(dataset, header=False, index=False)

if __name__ == '__main__':
    frame=np.zeros((64,64,3))
    trainx=np.zeros((3600,64,64,3))
    trainy=np.zeros((3600,),dtype=int)
    id=np.arange(3600)
    print(id)

    cnt=0
    print(trainy[0])
    for directory, subdirectories, files in os.walk(root):
        #print(subdirectories)
        #print(files)
        for file in files:
        # read the image file and extract its pixels
            im = imread(os.path.join(directory,file))
            frame[:,:,0]=im
            frame[:,:,1]=im
            frame[:,:,2]=im
            trainx[cnt,...]=frame
            trainy[cnt]=cnt//300
            cnt+=1
            if cnt%300==0:
                print(cnt/300)
    print(trainy)
    np.random.shuffle(id)
    trainx=trainx[id,...]
    trainy=trainy[id, ...]
    f.create_dataset("test_set_x", data=trainx)
    f.create_dataset("test_set_y", data=trainy)
    classes=np.arange(12)
    f.create_dataset("list_classes",data=classes)
    f.close()
    print(trainy)

import os
import cv2
import csv
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from scipy.cluster.vq import *
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import glob
import re
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import tree

labels = []
i=0
global_descriptor = np.array([], dtype="float32")
with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (i>5000):#limit for testing purpose ~70% of training data
            break 
        labels.append(row['Label'])
        #load image
        img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'],0)
        img = cv2.resize(img, (400, 250)) 
        print "Image: {}".format(row['Image'])
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(img, None)
        desc = np.squeeze(np.reshape(desc[:30], (1, -1)))
        global_descriptor = np.append(global_descriptor, desc)
        i+=1
    num_images=len(labels)
    print(num_images)
    global_descriptor = np.reshape(global_descriptor,(num_images,-1))
    print len(global_descriptor)

# clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
# clf.fit(global_descriptor,labels)

# clf = LogisticRegression()
# clf.fit(global_descriptor, labels)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(global_descriptor, labels)

test_labels = []
test_output=[]
j=-1

with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(j)
        j+=1
        if (j>5000):#Test last ~30% of training images
            test_labels.append(row['Label'])
            img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'],0)
            img = cv2.resize(img, (400, 250)) 

            sift = cv2.xfeatures2d.SIFT_create()
            kp, desc = sift.detectAndCompute(img, None)
            desc = desc[:30]
            desc = np.reshape(desc, (1,-1))
            label = clf.predict(desc)

            data = [[row['Image'].split('/')[-1], label[0]]]
            test_output.append(label[0])
            print(label)
    num_images=len(labels)

print(len(test_labels))
print(len(test_output))
report = classification_report(test_labels, test_output)
print '\nAccuracy of the model is: ' + str(accuracy_score(test_labels, test_output)) + "\n"
print 'Classification report: \n\n' + report
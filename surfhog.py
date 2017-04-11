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
from sklearn import tree

#Hog 30 scored better than hog 50
#Surf+Hog 30 descriptors with SVM scored highest 

labels = []
#i=0
global_descriptor = np.array([], dtype="float32")
with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #if (i>100):
        #    break 
        labels.append(row['Label'])
        img = cv2.imread('/Users/cimeajouz/Desktop/X_Train/'+ row['Image'],0)
        img = cv2.resize(img, (400, 250)) 
        print "Image: {}".format(row['Image'])
        surf = cv2.xfeatures2d.SURF_create()
        kp, surf_desc = surf.detectAndCompute(img, None)
        surf_desc = np.squeeze(np.reshape(surf_desc[:30], (1, -1)))
        hog = cv2.HOGDescriptor()
        hog_desc = hog.compute(img)
        hog_desc =np.squeeze(np.reshape(hog_desc[:30], (1, -1)))
        global_descriptor = np.append(global_descriptor, np.append(surf_desc, hog_desc))
        #i+=1
    num_images=len(labels)
    print(num_images)
    global_descriptor = np.reshape(global_descriptor,(num_images,-1))
    print len(global_descriptor)

#classifier = LogisticRegression()
#classifier.fit(global_descriptor, labels)
clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(global_descriptor,labels)

test_output=[]
with open('test_svm.csv', 'wb') as fp:
    writer = csv.writer(fp, delimiter=',')
    data = [['Image', 'Label']]
    writer.writerows(data)
    #natural_keys = []
    sorted_imgs=[]
    for img in glob.glob("/Users/cimeajouz/Desktop/X_Test/*.jpg"):
        sorted_imgs.append(int(img.split('/')[-1].split('.')[0]))
    sorted_imgs.sort()
    for image in sorted_imgs:
        img = cv2.imread('/Users/cimeajouz/Desktop/X_Test/'+str(image)+'.jpg', 0)
        img = cv2.resize(img, (400, 250))
        surf = cv2.xfeatures2d.SURF_create()
        kp, surf_desc = surf.detectAndCompute(img, None)
        surf_desc = surf_desc[:30]
        surf_desc = np.reshape(surf_desc, (1,-1))
        hog = cv2.HOGDescriptor()
        h = hog.compute(img)
        hog_desc = h[:30]
        hog_desc = np.reshape(hog_desc, (1,-1))
        desc = np.append(surf_desc, hog_desc)
        desc = np.reshape(desc,(1,-1))
        label = clf.predict(desc)
        data = [[str(image)+'.jpg', label[0]]]
        test_output.append(label[0])
        writer.writerows(data)
        print(clf.predict(desc))
    # report = classification_report(test_labels, test_output)
    # print '\nAccuracy of the model is: ' + str(accuracy_score(test_labels, test_output)) + "\n"
    # print 'Classification report: \n\n' + report

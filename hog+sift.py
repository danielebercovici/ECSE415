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

labels = []
i=0
global_descriptor = np.array([], dtype="float32")
with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (i>100):#limit for testing purpose ~70% of training data
            break 
        labels.append(row['Label'])
        #load image
        img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'],0)
        img = cv2.resize(img, (400, 250)) 
        print "Image: {}".format(row['Image'])
        hog = cv2.HOGDescriptor()
        hog_desc = hog.compute(img)
        hog_desc =np.squeeze(np.reshape(hog_desc[:30], (1, -1)))
        #print "hog_desc: {}".format(hog_desc)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, sift_desc = sift.detectAndCompute(img, None)
        sift_desc = np.squeeze(np.reshape(sift_desc[:30], (1, -1)))
        # print "sift_desc: {}".format(sift_desc)
        # print "len sift_desc: {}".format(len(sift_desc))
        global_descriptor = np.append(global_descriptor, np.append(sift_desc, hog_desc))
        i+=1
    num_images=len(labels)
    print(num_images)
    global_descriptor = np.reshape(global_descriptor,(num_images,-1))
    #print global_descriptor
    print len(global_descriptor)

clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
clf.fit(global_descriptor,labels)




test_labels = []
test_output=[]
j=-1
global_descriptor = np.array([], dtype="float32")

with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(j)
        j+=1
        if (j>5000):#Test last ~30% of training images
            test_labels.append(row['Label'])
            #load image
            img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'],0)
            img = cv2.resize(img, (400, 250)) 

            sift = cv2.xfeatures2d.SIFT_create()
            kp, sift_desc = sift.detectAndCompute(img, None)
            sift_desc = sift_desc[:30]
            sift_desc = np.reshape(sift_desc, (1,-1))

            hog = cv2.HOGDescriptor()
            h = hog.compute(img)
            hog_desc = h[:30]
            hog_desc = np.reshape(hog_desc, (1,-1))

            desc = np.append(sift_desc, hog_desc)
            #print(desc)
            desc = np.reshape(desc,(1,-1))
            label = clf.predict(desc)

            data = [[row['Image'].split('/')[-1], label[0]]]
            test_output.append(label[0])
            #writer.writerows(data)
            print(label)
    num_images=len(labels)
    global_descriptor = np.reshape(global_descriptor,(num_images,-1))
    #print global_descriptor
    #print len(global_descriptor)


# test_labels = []
# with open('Y_test_example.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         test_labels.append(row['Label'])
# #TEST
# test_output=[]
# with open('test.csv', 'wb') as fp:
#     writer = csv.writer(fp, delimiter=',')
#     data = [['Image', 'Label']]
#     writer.writerows(data)
#     for image in glob.glob("/Users/danielebercovici/Desktop/X_Test/*.jpg"):
#         print image.split('/')[-1]
#         img = cv2.imread(image, 0)
#         img = cv2.resize(img, (400, 250))
#         sift = cv2.xfeatures2d.SIFT_create()
#         kp, sift_desc = sift.detectAndCompute(img, None)
#         sift_desc = sift_desc[:50]
#         sift_desc = np.reshape(sift_desc, (1,-1))

#         hog = cv2.HOGDescriptor()
#         h = hog.compute(img)
#         hog_desc = h[:50]
#         hog_desc = np.reshape(hog_desc, (1,-1))

#         desc = np.append(sift_desc, hog_desc)
#         #print(desc)
#         desc = np.reshape(desc,(1,-1))
#         label = clf.predict(desc)

#         data = [[image.split('/')[-1], label[0]]]
#         test_output.append(label[0])
#         writer.writerows(data)
#         print(label)
    
    # report = classification_report(test_labels, test_output)
    # print '\nAccuracy of the model is: ' + str(accuracy_score(test_labels, test_output)) + "\n"
    # print 'Classification report: \n\n' + report
print(len(test_labels))
print(len(test_output))
report = classification_report(test_labels, test_output)
print '\nAccuracy of the model is: ' + str(accuracy_score(test_labels, test_output)) + "\n"
print 'Classification report: \n\n' + report




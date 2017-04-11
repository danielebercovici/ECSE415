import os
import cv2
import csv
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from scipy.cluster.vq import *
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import glob
import re
import seaborn as sns
from skimage import feature
from skimage import exposure
from sklearn import tree

#Hog 30 scored better than hog 50
#Surf+Hog 30 descriptors with SVM scored highest 

labels = []
i=0
global_descriptor = np.array([], dtype="float32")
with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (i>1):
           break 
        
        img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'],0)
        #img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'])
        img = cv2.resize(img, (400, 250)) 
        print "Image: {}".format(row['Image'])
        surf = cv2.xfeatures2d.SURF_create()
########forground extraction
        # mask = np.zeros(img.shape[:2],np.uint8)

        # bgdModel = np.zeros((1,65),np.float64)
        # fgdModel = np.zeros((1,65),np.float64)

        # rect = (5,5,img.shape[1]-10,img.shape[0]-10)
        # cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

        # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # img = img*mask2[:,:,np.newaxis]
        # img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
#############


        kp, surf_desc = surf.detectAndCompute(img, None)
        surf_desc = np.squeeze(np.reshape(surf_desc[:30], (1, -1)))

        #plot surf
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()

        hog = cv2.HOGDescriptor()
        hog_desc = hog.compute(img)
        #plot hog
        H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
        (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
 
        cv2.imshow("HOG Image", hogImage)
        
        hog_desc =np.squeeze(np.reshape(hog_desc[:400], (1, -1)))
        global_descriptor = np.append(global_descriptor, np.append(surf_desc, hog_desc))
        i+=1
    num_images=len(labels)
    print(num_images)
    global_descriptor = np.reshape(global_descriptor,(num_images,-1))
    print len(global_descriptor)

clf = LogisticRegression()
clf.fit(global_descriptor, labels)

# clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
# clf.fit(global_descriptor,labels)

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(global_descriptor, labels)

# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(global_descriptor, labels) 

test_labels = []
test_output=[]
j=-1

with open('Y_Train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(j)
        j+=1
        if (j>5000):#Test last ~30% of training images
            
            img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'],0)
            #img = cv2.imread('/Users/danielebercovici/Desktop/X_Train/'+ row['Image'])
            img = cv2.resize(img, (400, 250))
# ########forground extraction
#             mask = np.zeros(img.shape[:2],np.uint8)

#             bgdModel = np.zeros((1,65),np.float64)
#             fgdModel = np.zeros((1,65),np.float64)

#             rect = (5,5,img.shape[1]-10,img.shape[0]-10)
#             cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

#             mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#             img = img*mask2[:,:,np.newaxis]
#             img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
# #############  


            test_labels.append(row['Label'])          
            surf = cv2.xfeatures2d.SURF_create()
            kp, surf_desc = surf.detectAndCompute(img, None)
            
            surf_desc = surf_desc[:30]
            surf_desc = np.reshape(surf_desc, (1,-1))
            hog = cv2.HOGDescriptor()
            h = hog.compute(img)
            
            hog_desc = h[:400]
            hog_desc = np.reshape(hog_desc, (1,-1))
            desc = np.append(surf_desc, hog_desc)
            desc = np.reshape(desc,(1,-1))
            label = clf.predict(desc)
            #data = [[str(image)+'.jpg', label[0]]]
            test_output.append(label[0])
            print(label[0])
    num_images=len(labels)

print(len(test_labels))
print(len(test_output))
report = classification_report(test_labels, test_output)
print '\nAccuracy of the model is: ' + str(accuracy_score(test_labels, test_output)) + "\n"
print 'Classification report: \n\n' + report





# test_output=[]
# with open('test_svm.csv', 'wb') as fp:
#     writer = csv.writer(fp, delimiter=',')
#     data = [['Image', 'Label']]
#     writer.writerows(data)
#     #natural_keys = []
#     sorted_imgs=[]
#     for img in glob.glob("/Users/danielebercovici/Desktop/X_Test/*.jpg"):
#         sorted_imgs.append(int(img.split('/')[-1].split('.')[0]))
#     sorted_imgs.sort()
#     for image in sorted_imgs:
#         img = cv2.imread('/Users/danielebercovici/Desktop/X_Test/'+str(image)+'.jpg', 0)
#         #img = cv2.resize(img, (400, 250))
#         surf = cv2.xfeatures2d.SURF_create()
#         kp, surf_desc = surf.detectAndCompute(img, None)
#         surf_desc = surf_desc[:30]
#         surf_desc = np.reshape(surf_desc, (1,-1))
#         hog = cv2.HOGDescriptor()
#         h = hog.compute(img)
#         hog_desc = h[:400]
#         hog_desc = np.reshape(hog_desc, (1,-1))
#         desc = np.append(surf_desc, hog_desc)
#         desc = np.reshape(desc,(1,-1))
#         label = clf.predict(desc)
#         print(label)
#         data = [[str(image)+'.jpg', label[0]]]
#         test_output.append(label[0])
#         writer.writerows(data)
#         print(clf.predict(desc))




    # report = classification_report(test_labels, test_output)
    # print '\nAccuracy of the model is: ' + str(accuracy_score(test_labels, test_output)) + "\n"
    # print 'Classification report: \n\n' + report

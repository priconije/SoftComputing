from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter

# Train a multi-class linear SVM, courtesy of: http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
dataset = datasets.fetch_mldata("MNIST Original")

features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print "Count of digits in dataset", Counter(labels)

clf = LinearSVC()

clf.fit(hog_features, labels)

joblib.dump(clf, "digits_cls.pkl", compress=3)

from datetime import datetime
import os
import pickle
import random
import time
import warnings
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from descriptor import Descriptor


def processFiles(pos_dir, neg_dir, recurse=False, output_file=False,
        output_filename=None, color_space="bgr", channels=[0, 1, 2],
        hog_features=False, hist_features=False, spatial_features=False,
        hog_lib="cv", size=(64,64), hog_bins=9, pix_per_cell=(8,8),
        cells_per_block=(2,2), block_stride=None, block_norm="L1",
        transform_sqrt=True, signed_gradient=False, hist_bins=16,
        spatial_size=(16,16)):

    """
    Extract features from positive samples and negative samples.
    Store feature vectors in a dict and optionally save to pickle file.

    @param pos_dir (str): Path to directory containing positive samples.
    @param neg_dir (str): Path to directory containing negative samples.
    @param recurse (bool): Traverse directories recursively (else, top-level only).
    @param output_file (bool): Save processed samples to file.
    @param output_filename (str): Output file filename.
    @param color_space (str): Color space conversion.
    @param channels (list): Image channel indices to use.
    
    For remaining arguments, refer to Descriptor class:
    @see descriptor.Descriptor#__init__(...)

    @return feature_data (dict): Lists of sample features split into training,
        validation, test sets; scaler object; parameters used to
        construct descriptor and process images.

    NOTE: OpenCV HOGDescriptor currently only supports 1-channel and 3-channel
    images, not 2-channel images.
    """

    if not (hog_features or hist_features or spatial_features):
        raise RuntimeError("No features selected (set hog_features=True, "
            + "hist_features=True, and/or spatial_features=True.)")

    pos_dir = os.path.abspath(pos_dir)
    neg_dir = os.path.abspath(neg_dir)

    if not os.path.isdir(pos_dir):
        raise FileNotFoundError("Directory " + pos_dir + " does not exist.")
    if not os.path.isdir(neg_dir):
        raise FileNotFoundError("Directory " + neg_dir + " does not exist.")

    print("Building file list...")
    if recurse:
        pos_files = [os.path.join(rootdir, file) for rootdir, _, files
            in os.walk(pos_dir) for file in files]
        neg_files = [os.path.join(rootdir, file) for rootdir, _, files
            in os.walk(neg_dir) for file in files]
    else:
        pos_files = [os.path.join(pos_dir, file) for file in
            os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, file))]
        neg_files = [os.path.join(neg_dir, file) for file in
            os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, file))]

    print("{} positive files and {} negative files found.\n".format(
        len(pos_files), len(neg_files)))

    # Get color space information.
    color_space = color_space.lower()
    if color_space == "gray":
        color_space_name = "grayscale"
        cv_color_const = cv2.COLOR_BGR2GRAY
        channels = [0]
    elif color_space == "hls":
        color_space_name = "HLS"
        cv_color_const = cv2.COLOR_BGR2HLS
    elif color_space == "hsv":
        color_space_name = "HSV"
        cv_color_const = cv2.COLOR_BGR2HSV
    elif color_space == "lab":
        color_space_name = "Lab"
        cv_color_const = cv2.COLOR_BGR2Lab
    elif color_space == "luv":
        color_space_name = "Luv"
        cv_color_const = cv2.COLOR_BGR2Luv
    elif color_space == "ycrcb" or color_space == "ycc":
        color_space_name = "YCrCb"
        cv_color_const = cv2.COLOR_BGR2YCrCb
    elif color_space == "yuv":
        color_space_name = "YUV"
        cv_color_const = cv2.COLOR_BGR2YUV
    else:
        color_space_name = "BGR"
        cv_color_const = -1

    # Get names of desired features.
    features = [feature_name for feature_name, feature_bool
        in zip(["HOG", "color histogram", "spatial"],
               [hog_features, hist_features, spatial_features])
        if feature_bool == True]

    feature_str  = features[0]
    for feature_name in features[1:]:
        feature_str += ", " + feature_name

    # Get information about channel indices.
    if len(channels) == 2 and hog_features and hog_lib == "cv":
        warnings.warn("OpenCV HOG does not support 2-channel images",
            RuntimeWarning)

    channel_index_str = str(channels[0])
    for ch_index in channels[1:]:
        channel_index_str += ", {}".format(ch_index)

    print("Converting images to " + color_space_name + " color space and " 
        + "extracting " + feature_str + " features from channel(s) "
        + channel_index_str + ".\n")

    # Store feature vectors for positive samples in list pos_features and
    # for negative samples in neg_features.
    pos_features = []
    neg_features = []
    start_time = time.time()

    # Get feature descriptor object to call on each sample.
    descriptor = Descriptor(hog_features=hog_features, hist_features=hist_features,
            spatial_features=spatial_features, hog_lib=hog_lib, size=size,
            hog_bins=hog_bins, pix_per_cell=pix_per_cell,
            cells_per_block=cells_per_block, block_stride=block_stride,
            block_norm=block_norm, transform_sqrt=transform_sqrt,
            signed_gradient=signed_gradient, hist_bins=hist_bins,
            spatial_size=spatial_size)

    # Iterate through files and extract features.
    for i, filepath in enumerate(pos_files + neg_files):
        image = cv2.imread(filepath)
        if cv_color_const > -1:
            image = cv2.cvtColor(image, cv_color_const)

        if len(image.shape) > 2:
            image = image[:,:,channels]

        feature_vector = descriptor.getFeatureVector(image)

        if i < len(pos_files):
            pos_features.append(feature_vector)
        else:
            neg_features.append(feature_vector)

    print("Features extracted from {} files in {:.1f} seconds\n".format(
        len(pos_features) + len(neg_features), time.time() - start_time))

    # Store the length of the feature vector produced by the descriptor.
    num_features = len(pos_features[0])
    #print(len(pos_features)) # 8792
    #print(pos_features[0].shape) #1764
    #print(len(neg_features)) # 8968
    #print(neg_features[0].shape) # 1764

    ##TODO: Instantiate scaler and scale features.
    scaler = StandardScaler()
    scaler.fit(pos_features+neg_features)
    pos_features = scaler.transform(pos_features)
    neg_features = scaler.transform(neg_features)

    ##TODO: Randomize lists of feature vectors. Split 75/20/5 into training,
    # validation, and test sets.
    print("Shuffling samples into training, cross-validation, and test sets.\n")
    random.shuffle(pos_features)
    random.shuffle(neg_features)

    # Use pos_train, pos_val, pos_test and neg_train, neg_val, neg_test to represent 
	# the Train, Validation and Test sets of Positive and Negtive sets.
    from sklearn.model_selection import train_test_split
    pos_train, pos_test = train_test_split(pos_features,test_size = 0.05,random_state = 42)
    pos_train, pos_val= train_test_split(pos_train,test_size = 20.0/95,random_state = 42)

    neg_train, neg_test = train_test_split(neg_features,test_size = 0.05,random_state = 42)
    neg_train, neg_val= train_test_split(neg_train,test_size = 20.0/95,random_state = 42)
    # Store sample data and parameters in dict.
    # Descriptor class object seems to produce errors when unpickling and
    # has been commented out below. The descriptor will be re-instantiated
    # by the Detector object later.
    feature_data = {
                    "pos_train": pos_train,
                    "neg_train": neg_train,
                    "pos_val": pos_val,
                    "neg_val": neg_val,
                    "pos_test": pos_test,
                    "neg_test": neg_test,
                    #"descriptor": descriptor,
                    "scaler": scaler,
                    "hog_features": hog_features,
                    "hist_features": hist_features,
                    "spatial_features": spatial_features,
                    "color_space": color_space,
                    "cv_color_const": cv_color_const,
                    "channels": channels,
                    "hog_lib": hog_lib,
                    "size": size,
                    "hog_bins": hog_bins,
                    "pix_per_cell": pix_per_cell,
                    "cells_per_block": cells_per_block,
                    "block_stride": block_stride,
                    "block_norm": block_norm,
                    "transform_sqrt": transform_sqrt,
                    "signed_gradient": signed_gradient,
                    "hist_bins": hist_bins,
                    "spatial_size": spatial_size,
                    "num_features": num_features
                    }

    # Pickle to file if desired.
    if output_file:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M")
                + "_data.pkl")

        pickle.dump(feature_data, open(output_filename, "wb"))
        print("Sample and parameter data saved to {}\n".format(output_filename))

    return feature_data

def trainSVM(filepath=None, feature_data=None, C=1,
        loss="squared_hinge", penalty="l2", dual=False, fit_intercept=False,
        output_file=False, output_filename=None):

    """
        Train a classifier from feature data extracted by processFiles().

        @param filepath (str): Path to feature data pickle file.
        @param feature_data (dict): Feature data dict returned by processFiles().
            NOTE: Either a file or dict may be supplied.
        @param output_file (bool): Save classifier and parameters to file.
        @param output_filename (str): Name of output file.

        For remaining arguments, @see sklearn.svm.LinearSVC()

        @return classifier_data (dict): Dict containing trained classifier and
            relevant training/processing feature parameters.
    """

    print("Loading sample data.")
    if filepath is not None:
        filepath = os.path.abspath(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File " + filepath + " does not exist.")
        feature_data = pickle.load(open(filepath, "rb"))
    elif feature_data is None:
        raise ValueError("Invalid feature data supplied.")

    ##TODO: Train classifier on training set, using sklearn LinearSVC model. 
	##      Use validation sets to adjust your algorithm. 
	##      Run your classifier on the test sets and output the accuracy, 
	##      precission, recall and F-1 score.
    import numpy as np
    pos_train = feature_data["pos_train"]  
    pos_train_y = np.ones(len(pos_train),) 
    neg_train = feature_data["neg_train"]
    neg_train_y = np.zeros(len(neg_train),) 
    x_train = np.vstack((pos_train,neg_train))   
    y_train = np.concatenate((pos_train_y,neg_train_y))

    pos_val = feature_data["pos_val"]
    pos_val_y = np.ones(len(pos_val),) 
    neg_val = feature_data["neg_val"]
    neg_val_y = np.zeros(len(neg_val),) 
    x_val = np.vstack((pos_val,neg_val))   
    y_val = np.concatenate((pos_val_y,neg_val_y))

    pos_test = feature_data["pos_test"]
    pos_test_y = np.ones(len(pos_test),)
    neg_test = feature_data["neg_test"]
    neg_test_y = np.zeros(len(neg_test),)
    x_test = np.vstack((pos_test,neg_test))
    y_test = np.concatenate((pos_test_y,neg_test_y))
    
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix, accuracy_score
    svc = LinearSVC(penalty=penalty,C=C,loss=loss,dual=dual,fit_intercept=fit_intercept)
    svc.fit(x_train,y_train)
    y_val_pred = svc.predict(x_val)
    tn1,fp1,fn1,tp1 = confusion_matrix(y_val, y_val_pred).ravel()
    accuracy1 = (tp1 + tn1)/(tp1 + fp1 + fn1 + tn1)
    precision1 = tp1/(tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)
    f11 = 2 * (precision1 * recall1)/(precision1 + recall1)

    print("Before adjustment, Accuracy:{}, Precision:{},Recall:{},F-1:{}".format(accuracy1,precision1,recall1,f11))

    x_train_two = np.vstack([x_train,x_val])
    y_train_two = np.concatenate((y_train,y_val))
    svc.fit(x_train_two,y_train_two)
    y_pred = svc.predict(x_test)
    
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn)/(tp + fp + fn + tn)
    precision = tp/(tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall)/(precision + recall)
    print("After adjustment, Accuracy:{}, Precision:{},Recall:{},F-1:{}".format(accuracy,precision,recall,f1))


    # Store classifier data and parameters in new dict that excludes
    # sample data from feature_data dict.
    excludeKeys = ("pos_train", "neg_train", "pos_val", "neg_val","pos_test", "neg_test")
    classifier_data = {key: val for key, val in feature_data.items() if key not in excludeKeys}
    classifier_data["classifier"] = svc

    if output_file:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M")
                + "_classifier.pkl")

        pickle.dump(classifier_data, open(output_filename, "wb"))
        print("\nSVM classifier data saved to {}".format(output_filename))

    return classifier_data

import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "Desktop/ML/lab7/Lab07/samples/vehicles"
neg_dir = "Desktop/ML/lab7/Lab07/samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "Desktop/ML/lab7/Lab07/videos/test_video.mp4"

def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """

    # Extract HOG features from images in the sample directories and return
    # results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
        hog_features=True)

    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)
	

	##TODO: If you have trained your classifier and prepare to detect the video, 
	##      uncomment the code below.
	
    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)
    
    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap,write = True)


def experiment2():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """

    # Extract HOG features from images in the sample directories and return
    # results and parameters in a dict.
    color_space="hsv"
    output_dir = "Desktop/ML/lab7/Lab07/output/"
    output_filename = output_dir+"Experiment2, color_space = hsv"+ "feature_data.pkl"
    feature_data = processFiles(pos_dir, neg_dir,output_file = True,color_space=color_space, hist_features=True, 
    spatial_features=True, recurse=True,hog_features=True,output_filename=output_filename)
    
    output_filename_train=output_dir+"Experiment2"+ "svm_model.pkl"
    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(output_filename=output_filename_train,output_file=True,feature_data=feature_data)
	

	##TODO: If you have trained your classifier and prepare to detect the video, 
	##      uncomment the code below.
    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)
    
    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap,write = True)
def experiment3():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """

    # Extract HOG features from images in the sample directories and return
    # results and parameters in a dict.
    color_space="yuv"
    output_dir = "Desktop/ML/lab7/Lab07/output/"
    output_filename = output_dir+"Experiment3,color_space = yuv"+ "feature_data.pkl"
    feature_data = processFiles(pos_dir, neg_dir,output_file = True,color_space=color_space, hist_features=False, 
    spatial_features=True, recurse=True,hog_features=True,output_filename=output_filename,block_norm="L2", transform_sqrt=False,
        spatial_size=(64, 32))
    
    output_filename_train=output_dir+"Experiment3"+ "svm_model.pkl"
    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(loss="squared_hinge",penalty="l2", dual=False, fit_intercept=False,output_filename=output_filename_train,output_file=True,feature_data=feature_data)
	

	##TODO: If you have trained your classifier and prepare to detect the video, 
	##      uncomment the code below.
    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector(init_size=(64,64), x_overlap=0.3, y_step=0.015,
        x_range=(0.1, 0.9), scale=1.4).loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)
    
    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap, num_frames=5, threshold=100,
        min_bbox=(50,50), draw_heatmap=False)
def experiment4():
    """
    Train a classifier and run on video using parameters that seemed to work
    well for vehicle detection.
    """
    color_space="ycrcb"
    output_dir = "Desktop/ML/lab7/Lab07/output/"
    output_filename = output_dir+"Experiment4,color_space = ycrcb"+ "feature_data.pkl"
    feature_data = processFiles(pos_dir, neg_dir,output_file = True,output_filename=output_filename,recurse=True,
        color_space=color_space, channels=[0, 1, 2], hog_features=True,
        hist_features=True, spatial_features=True, hog_lib="cv",
        size=(64,64), pix_per_cell=(8,8), cells_per_block=(2,2),
        hog_bins=15, hist_bins=16, spatial_size=(20,20))

    output_filename_train=output_dir+"Experiment4"+ "svm_model.pkl"

    classifier_data = trainSVM(feature_data=feature_data, C=1000,output_filename=output_filename_train,output_file=True)

    detector = Detector(init_size=(90,90), x_overlap=0.7, y_step=0.01,
        x_range=(0.02, 0.98), y_range=(0.55, 0.89), scale=1.3)
    detector.loadClassifier(classifier_data=classifier_data)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap,write = True, num_frames=9, threshold=100,
        draw_heatmap_size=0.3)  
if __name__ == "__main__":
    #experiment1()
    #experiment2() 
	#experiment3()
    experiment4()

import glob
import numpy as np
import os
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_reader import DataReader
from preprocessor import Preprocessor


class ConfigureData:
    # Folder paths to the database with training and testing data samples
    BASE_FOLDER = r'/home/iulia/dev/Hand Gesture Database'
    TRAIN_FOLDER = BASE_FOLDER + r'/train'
    TEST_FOLDER = BASE_FOLDER + r'/test'

    # Vision sensor detection range
    POSITION_MAX = [350, 750, 350]
    POSITION_MIN = [-350, 50, -350]
    VECTOR_MAX = [1,1,1]
    VECTOR_MIN = [-1,-1,-1]


    def load_data(data_reader: DataReader, data_paths: list) -> list[list[float]]:
        """
        Load gesture data from specified file paths.

        Parameters:
        - data_reader: An instance of DataReader to parse the data.
        - data_paths: List of file paths to the gesture data.

        Returns:
        - gesture_data: A list containing lists of sequential frames for each gesture.
        """
        gesture_data = []
        if len(data_paths) == 0:
            raise FileNotFoundError(f"Database files have not been found. Check the folder file path.")
        for path in data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            sequential_frames = data_reader.load_data(path)
            gesture_data.append(sequential_frames)
        return gesture_data
    
    def load_labels(data_path: list, gesture_names: list) -> np.ndarray:
        """
        Generate labels for the dataset based on the gesture names.

        Parameters:
        - data_paths: list of file paths
        - gesture_names: list of gesture names corresponding to labels

        Returns:
        - labels: np.array of shape (n_samples,) with integer labels
        """
        labels = np.zeros((len(data_path),))
        count_pos = 0
        for i in range(0, len(gesture_names)):
            for j in range(0, round(len(data_path)/len(gesture_names))):
                labels[count_pos,] = i
                count_pos = count_pos + 1
        return labels
    
    def extract_features(data: list[list[float]], preprocessor: Preprocessor) -> np.ndarray:
        """
            Extract and normalize features from gesture data.

            Parameters:
            - data: raw gesture data where the outer list is a list of all samples, and the inner list is a list of frames for each sample
            - preprocessor: Instance of Preprocessor for feature extraction.

            Returns:
            - x_data: Normalized feature vectors for each gesture.
        """
        extract_training_data = preprocessor.raw_data_extraction(data, ConfigureData.POSITION_MAX,ConfigureData.POSITION_MIN, ConfigureData.VECTOR_MAX, ConfigureData.VECTOR_MIN)
        norm_data = preprocessor.normalising_data_01(extract_training_data[0], extract_training_data[1], extract_training_data[2])
        x_data = preprocessor.feature_extraction(norm_data, 15, 5)
        return x_data
    
    def print_results(y_test: np.ndarray, y_pred: np.ndarray):
        """
        Print classification metrics.

        Parameters:
        - y_test: true labels
        - y_pred: predicted labels
        """
        print(f"Accuracy: {sum(y_test==y_pred)/y_test.shape[0]}")
        
        # calculate and print macro-averaged precision, recall, and F1 score
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        print('Macro-averaged precision: ', precision)
        print('Macro-averaged recall: ', recall)
        print('Macro-averaged F1 score: ', macro_f1)

    def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
        """
        Plot the confusion matrix.

        Parameters:
        - y_test: true labels
        - y_pred: predicted labels
        - gesture_names: list of gesture class numbers
        """
        conf_matrix = confusion_matrix(y_test, y_pred)

        gesture_numbers = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14']
        
        conf_df = pd.DataFrame(conf_matrix, index=gesture_numbers, columns=gesture_numbers)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, annot_kws={"size": 16})
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.show()



# Main code
if __name__ == "__main__":
    # Gesture class names
    gesture_names = [
        'forward', 'backwards', 'left', 'right', 'up', 'down', 
        'grasp', 'release', 'rotateposx', 'rotatenegx', 
        'rotateposy', 'rotatenegy', 'rotateposz', 'rotatenegz'
    ]
    
    # Paths for training and testing data
    folder_training_path = ConfigureData.TRAIN_FOLDER
    folder_testing_path = ConfigureData.TEST_FOLDER
    training_path = []
    testing_path = []

    for i in range(len(gesture_names)):
        training_load = folder_training_path + '/' + gesture_names[i] + '/*.csv'
        testing_load = folder_testing_path + '/' + gesture_names[i] + '/*.csv'
        training_path = training_path + glob.glob(training_load)
        testing_path = testing_path + glob.glob(testing_load)


    data_reader = DataReader()
    preprocessor = Preprocessor()

    # Load training and test data
    gesture_training_data = ConfigureData.load_data(data_reader, training_path)
    gesture_testing_data = ConfigureData.load_data(data_reader, testing_path)

    gesture_training_data = np.array(gesture_training_data)
    gesture_testing_data = np.array(gesture_testing_data)

    # Load labels
    y_train = ConfigureData.load_labels(training_path, gesture_names)
    y_test = ConfigureData.load_labels(testing_path, gesture_names)

    # Feature Extraction
    x_train = ConfigureData.extract_features(gesture_training_data, preprocessor)
    x_test = ConfigureData.extract_features(gesture_testing_data, preprocessor)

    # Train and evaluate the classifier (multi-class classification one-versus-rest approach)
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', C=5, gamma=0.1))
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    # Save the trained model
    joblib.dump(classifier, 'svm_model.joblib')

    # Print results and plot confusion matrix
    ConfigureData.print_results(y_test, y_pred)
    ConfigureData.plot_confusion_matrix(y_test, y_pred)

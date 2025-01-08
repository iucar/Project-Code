import math
import numpy as np

NO_FRAMES = 75

class Preprocessor:
    def __init__(self):
        """
        Class for preprocessing raw data into a format suitable for feature extraction and normalisation.
        """
        pass

    # Function that extracts the chosen positional data and concatenates it into a vector
    def raw_data_extraction(self, data_samples: np.array, position_max: list[int], position_min: list[int], vector_max: list[int], vector_min: list[int]) -> tuple[np.array, np.array, np.array]:
        """
        Extract raw data and normalize bounds for positional tracking data.

        Parameters:
        - data_samples: np.ndarray of shape (n_samples, n_frames, frame_data_structure)
        - position_max, position_min, vector_max, vector_min: list of integers defining normalization bounds in each axis [x,y,z]

        Returns:
        - formatted_data: np.ndarray of shape (n_samples, n_frames, n_features)
        - normalising_max: np.ndarray for normalization max values
        - normalising_min: np.ndarray for normalization min values
        """
        # For each frame in each sample get the positional tracking data
        formatted_data = np.array([
            np.array([self._get_tracking_data(frame) for frame in sample]) 
            for sample in data_samples
        ])

        # Normalisation bounds based on the type of data
        normalising_max = np.array(position_max * 5 + vector_max * 2)
        normalising_min = np.array(position_min * 5 + vector_min * 2)

        return formatted_data, normalising_max, normalising_min

    def _get_tracking_data(self, frame) -> np.array:
        """
        Extract and concatenate tracking data for a single frame.

        Parameters:
        - frame: a frame object providing access to positional tracking data.

        Returns:
        - tracking_data: np.ndarray containing concatenated positional data.
        """
    
        tracking_data = np.concatenate([
            np.array(frame.get_palm_position()),
            np.array(frame.get_next_joint_bone(1, 3)),
            np.array(frame.get_next_joint_bone(2, 3)),
            np.array(frame.get_next_joint_bone(3, 3)),
            np.array(frame.get_next_joint_bone(4, 3)),
            np.array(frame.get_palm_normal()),
            np.array(frame.get_palm_direction())
        ])
        return tracking_data


    def normalising_data_01(self, data_samples_fv: np.array, normalising_max: np.array, normalising_min: np.array) -> np.array:
        """
        Normalize concatenated vector data to range [0, 1].

        Parameters:
        - data_samples_fv: np.ndarray containing feature vector data.
        - normalising_max: np.ndarray of max values for normalization.
        - normalising_min: np.ndarray of min values for normalization.

        Returns:
        - normalised_data: np.ndarray with normalized values.
        """

        normalised_data = (data_samples_fv - normalising_min) / (normalising_max - normalising_min)
        return normalised_data

    def feature_extraction(self, data_samples_fv: np.array, window_length: int, window_overlap: int) -> np.array:
        """
        Extract non-geometrical data features using a sliding window.
        For each window and each data element within the window calculate the following features:
            - mean
            - min value
            - max value
            - standard deviation
            - root mean square

        Parameters:
        - data_samples_fv: np.ndarray of shape (n_samples, n_frames, n_features).
        - window_length: number of frames in each sliding window.
        - window_overlap: overlap between consecutive windows.

        Returns:
        - formatted_data: np.ndarray of extracted features for each sample.
        """
        feature_length = int(((NO_FRAMES - window_length) / (window_length - window_overlap)) + 1)

        n_features = 5  # mean, min, max, sd, rms

        n_samples, n_frames, n_elements = data_samples_fv.shape
        feature_vector_len = n_features * n_elements * feature_length
        formatted_data = np.zeros((n_samples, feature_vector_len))


        for sample_idx, sample in enumerate(data_samples_fv):
            sample_feature_vector = []

            for i in range(feature_length):
                # Organize the sliding window by frames
                window_features = sample[
                    (i * (window_length - window_overlap)) : ((i * (window_length - window_overlap)) + window_length)
                ]
                # Organize by positional tracking data
                element_array = np.array(window_features).T  # Transpose to get features per dimension

                for element in element_array:
                    sample_feature_vector.extend([
                        np.mean(element),            # Mean
                        np.min(element),             # Min
                        np.max(element),             # Max
                        np.std(element),             # Standard Deviation
                        np.sqrt(np.mean(element**2)) # RMS
                    ])

            sample_feature_vector = np.array(sample_feature_vector)
            formatted_data[sample_idx, :] = sample_feature_vector

        return formatted_data

    def add_gaussian_noise(self, data_samples: np.array, mean: float, std: float) -> np.array:
        """
        Add Gaussian noise to data and clip to range [0, 1]. Can be used to create more data samples from
        existing samples to increase vareity in a limited dataset.

        Parameters:
        - data_samples: np.ndarray of shape (n_samples, n_frames, n_features).
        - mean: mean of the Gaussian noise.
        - std: standard deviation of the Gaussian noise.

        Returns:
        - noisy_data: np.ndarray with added Gaussian noise.
        """
        data_samples = np.array(data_samples)
        noisy_data = data_samples + np.random.normal(mean, std, data_samples.shape)
        noisy_data = np.clip(noisy_data, 0, 1)
        return noisy_data

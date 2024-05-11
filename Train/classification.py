import csv
import random
import joblib
import numpy as np
from itertools import islice
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


class FeatureClassification:
    def __init__(self, clonefeature_csv, nonclonefeature_csv):
        self.clonefeature_csv = clonefeature_csv
        self.nonclonefeature_csv = nonclonefeature_csv

    def feature_extraction_order(self, feature_csv):
        """
            Reads a CSV file containing numerical features and converts them into a list of lists,
            where each inner list contains float values of features for one sample.

            Args:
            feature_csv (str): The path to the CSV file from which to extract features.

            Returns:
            list: A list of lists, where each sublist represents the features of a single sample,
                  converted into floats.
        """
        features = []
        with open(feature_csv, 'r') as f:
            data = csv.reader(f)
            # Iterate over each row in the CSV file.
            for line in islice(data, 0, None):
                # Convert to float type.
                feature = [float(i) for i in line]
                features.append(feature)
            print(len(features))
            print(len(features[0]))
        return features

    def obtain_dataset_order(self):
        """
            Loads feature data from two specified CSV files, one for 'clone' samples and one for 'non-clone' samples,
            and constructs a dataset consisting of feature vectors and corresponding labels.

            Args:
            clonefeature_path (str): The file path to the CSV containing features for 'clone' samples.
            noclonefeature_path (str): The file path to the CSV containing features for 'non-clone' samples.

            Returns:
            tuple: A tuple containing two lists:
                   - Vectors: A list of feature vectors where each vector is a list of floats.
                   - Labels: A list of integers where each integer is a label (1 for 'clone', 0 for 'non-clone').
             """

        clone_features = self.feature_extraction_order(self.clonefeature_csv)
        nonclone_features = self.feature_extraction_order(self.nonclonefeature_csv)
        print('len:')
        print(len(clone_features))
        print(len(nonclone_features))
        print(len(clone_features[0]))
        print(len(nonclone_features[0]))

        Vectors = []
        Labels = []
        Vectors.extend(clone_features)
        Labels.extend([1 for _ in range(len(clone_features))])  # Set the clone label to 1.
        Vectors.extend(nonclone_features)
        Labels.extend([0 for _ in range(len(nonclone_features))])  # Set the clone label to 0.

        return Vectors, Labels

    def random_features_order(self, vectors, labels):
        """
            Combines feature vectors with their corresponding labels, shuffles the combined list randomly,
            and then separates them back into shuffled vectors and labels.

            Args:
            vectors (list of lists): A list where each element is a list representing a feature vector.
            labels (list): A list of labels corresponding to each feature vector.

            Returns:
            tuple: A tuple containing two elements:
                   - A list of feature vectors, shuffled and without their corresponding labels.
                   - A list of labels, shuffled in accordance with their feature vectors.
        """
        Vec_Lab = []
        for i in range(len(vectors)):
            vec = vectors[i]
            lab = labels[i]
            vec.append(lab)
            Vec_Lab.append(vec)

        random.shuffle(Vec_Lab)

        return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]

    def XGBOOST(self, X, Y):
        """
            Performs a 10-fold cross-validation on the given dataset using an XGBoost classifier,
            evaluates the model using F1 score, precision, and recall, and saves the best model based
            on the highest F1 score obtained.

            Args:
            X (array-like): Feature matrix where each row represents a sample and each column represents a feature.
            Y (array-like): Corresponding labels for the samples in X.

            Returns:
            list: A list containing the mean F1 score, mean precision, and mean recall from the 10 folds.
        """
        print("begin")
        kf = KFold(n_splits=10)
        F1s = []
        Precisions = []
        Recalls = []

        best_f1 = 0  # Initialize the highest F1 score
        best_model = None  # Initialize storage variable for the best model

        for train_index, test_index in kf.split(X):
            train_X, train_Y = X[train_index], Y[train_index]
            test_X, test_Y = X[test_index], Y[test_index]

            clf = XGBClassifier(max_depth=256, random_state=0)
            clf.fit(train_X, train_Y)

            y_pred = clf.predict(test_X)
            f1 = f1_score(y_true=test_Y, y_pred=y_pred)
            precision = precision_score(y_true=test_Y, y_pred=y_pred)
            recall = recall_score(y_true=test_Y, y_pred=y_pred)

            F1s.append(f1)
            Precisions.append(precision)
            Recalls.append(recall)

            # Check if it's the best model and update the highest F1 score and save the model
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf

        # Save the best-performing model
        if best_model is not None:
            joblib.dump(best_model, 'best_model.pkl')

        print('F1.Precision.Recalls:')
        print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
        return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]

    def run(self):
        Vectors, Labels = self.obtain_dataset_order()
        vectors, labels = self.random_features_order(Vectors, Labels)

        print('vectors:')
        print(len(vectors))
        print('labels:')
        print(len(labels))
        vectors = np.array(vectors)
        labels = np.array(labels)

        target = self.XGBOOST(vectors, labels)
        print(target)


if __name__ == '__main__':
    processor = FeatureClassification("BCB_clone_sample_4_dis.csv", "BCB_nonclone_sample_4_dis.csv")
    processor.run()
import os
import time
from get_matrix import JavaSyntaxMatrixGenerator
from get_distance import DistanceCalculator
from classification import FeatureClassification


class TrainSystem:
    def __init__(self, java_path, clone_path, nonclone_path, feature_number=400, npy_path='./npy/', json_path='type.json'):
        self.java_path = java_path
        self.clone_path = clone_path
        self.nonclone_path = nonclone_path
        self.npy_path = npy_path
        self.json_path = json_path
        self.feature_number = feature_number
        self.clone_feature_csv = os.path.splitext(os.path.basename(clone_path))[0]
        self.nonclone_feature_csv = os.path.splitext(os.path.basename(nonclone_path))[0]

    def prepare_matrices(self):
        print("Generating syntax matrices...")
        syntax_matrix_generator = JavaSyntaxMatrixGenerator(self.java_path, self.npy_path, self.json_path)
        start_time = time.time()
        syntax_matrix_generator.allmain()
        print("Matrix generation completed in {:.2f} seconds.".format(time.time() - start_time))

    def calculate_distances(self):
        print("Calculating distances...")
        start_time = time.time()
        distance_calculator = DistanceCalculator(self.clone_path, self.npy_path, self.feature_number)
        distance_calculator.get_distance()
        distance_calculator = DistanceCalculator(self.nonclone_path, self.npy_path, self.feature_number)
        distance_calculator.get_distance()
        print("Distance calculations completed in {:.2f} seconds.".format(time.time() - start_time))

    def train_classifier(self):
        print("Training classifier...")
        classifier = FeatureClassification(self.clone_feature_csv + '_4_dis.csv', self.nonclone_feature_csv + '_4_dis.csv')
        classifier.run()
        print("Classifier training completed.")

    def run(self):
        # Step 1: Generate the matrices
        self.prepare_matrices()

        # Step 2: Calculate distances between matrices
        self.calculate_distances()

        # Step 3: Train the classification model
        self.train_classifier()


if __name__ == "__main__":
    # Paths and file names need to be correctly set according to your project structure
    java_path = './BCB/'
    nonclone_path = './Clone_type/BCB_nonclone.csv'
    clone_path = './Clone_type/BCB_clone.csv'

    train_system = TrainSystem(java_path, clone_path, nonclone_path)
    train_system.run()

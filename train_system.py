import os
import time
from get_matrix import JavaSyntaxMatrixGenerator
from get_distance import DistanceCalculator
from classification import FeatureClassification


class TrainSystem:
    def __init__(self, java_path, csv_path, clone_csv, nonclone_csv, npy_path='./npy/', json_path='type.json'):
        self.java_path = java_path
        self.csv_path = csv_path
        self.clone_csv = clone_csv
        self.nonclone_csv = nonclone_csv
        self.npy_path = npy_path
        self.json_path = json_path

    def prepare_matrices(self):
        print("Generating syntax matrices...")
        syntax_matrix_generator = JavaSyntaxMatrixGenerator(self.java_path, self.npy_path, self.json_path)
        start_time = time.time()
        syntax_matrix_generator.allmain()
        print("Matrix generation completed in {:.2f} seconds.".format(time.time() - start_time))

    def calculate_distances(self):
        print("Calculating distances...")
        distance_calculator = DistanceCalculator(self.csv_path, self.npy_path)
        distance_calculator.get_distance()
        print("Distance calculations completed.")

    def train_classifier(self):
        print("Training classifier...")
        classifier = FeatureClassification(self.clone_csv, self.nonclone_csv)
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
    java_path = './BCB_datasets_samples'
    csv_path = './Clone_type/BCB_nonclone.csv'
    clone_csv = "BCB_clone_sample_4_dis.csv"
    nonclone_csv = "BCB_nonclone_sample_4_dis.csv"

    train_system = TrainSystem(java_path, csv_path, clone_csv, nonclone_csv)
    train_system.run()

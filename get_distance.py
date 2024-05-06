import numpy as np
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


class DistanceCalculator:
    def __init__(self, ori, out, npy_path, feature_number=400):
        self.ori = ori
        self.out = out
        self.npy_path = npy_path
        self.feature_number = feature_number

    def listdir(self, path):
        """
        Recursively lists all files in the specified directory and subdirectories.

        Args:
        path (str): The directory path to list files from.

        Returns:
        list: A list of all file paths accumulated.
           """
        javalist = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                javalist.extend(self.listdir(file_path))
            else:
                javalist.append(file_path)
        return javalist

    def get_distance(self):
        """
            Calculates and stores multiple distance metrics between pairs of files specified in a CSV file.
            This function processes each pair of file names in the CSV, checks if their corresponding .npy
            matrix files exist, and if so, calculates cosine similarity, Euclidean distance, Manhattan distance,
            and Chebyshev distance between these matrices.

            Args:
            ori (str): The path to the CSV file containing pairs of filenames to compare.
            out (str): The base name for the output CSV file where the distances will be stored.
            path (str): The directory path where the .npy files are located.

            Returns:
            None: This function writes results to a CSV file and does not return any value.
            """
        weightorder = []
        with open('weight.txt', 'r') as f:
            lines = f.readlines()[:self.feature_number]
            for line in lines:
                value = line.split()[1]
                weightorder.append(int(value))
        print(weightorder)
        print(len(weightorder))

        # Recursively list all .npy files in the given directory and subdirectories
        existnpy = self.listdir(self.npy_path)
        j = 0

        exc = []
        reader = csv.reader(open(self.ori, 'r'))

        for r in reader:
            try:
                # Ensure there are at least two columns per row
                if len(r) < 2:
                    continue  # Skip rows that do not have at least two filenames

                # Extract filenames from the first two columns of the CSV and strip the '.java' suffix safely
                f1 = r[0].rsplit('.java', 1)[0]  # Splits only on the last occurrence of '.java'
                f2 = r[1].rsplit('.java', 1)[0]

                # Construct the paths to the corresponding .npy files using os.path.join for better path handling
                file1 = os.path.join(self.npy_path, f1 + '.npy')
                file2 = os.path.join(self.npy_path, f2 + '.npy')
            except IndexError:
                print("Error processing row:", r)  # Log an error message if the row is not as expected
                continue  # Skip to the next row

            if file1 in existnpy and file2 in existnpy:

                # Load the .npy files if they exist
                matrix1 = np.load(file1)
                matrix2 = np.load(file2)

                # Calculate cosine similarity, Euclidean, Manhattan, and Chebyshev distances
                cos = cosine_similarity(matrix1, matrix2)
                euc = pairwise_distances(matrix1, matrix2)
                man = pairwise_distances(matrix1, matrix2, metric='manhattan')
                che = pairwise_distances(matrix1, matrix2, metric='chebyshev')

                cosine = []
                euclidean = []
                manhattan = []
                chebyshev = []

                # Extract diagonal elements which represent the distances between identical indices
                for i in range(len(che[0])):
                    cosine.append(1 - cos[i][i])
                    euclidean.append(euc[i][i])
                    manhattan.append(man[i][i])
                    chebyshev.append(che[i][i])

                data = cosine + euclidean + manhattan + chebyshev
                extracted_data = [data[i] for i in weightorder if i < len(data)]
                exc.append(extracted_data)  # Append the computed data to the list

                print(j)  # Print the current count
                j += 1

                if j > 1000:
                    # Write all computed distances to a new CSV file named based on the 'out' parameter
                    with open(self.out + '_4_dis.csv', 'w', newline='') as csvfile0:
                        writer = csv.writer(csvfile0)
                        for row in exc:
                            writer.writerow(row)
                    break


if __name__ == '__main__':
    calc = DistanceCalculator('./Clone_type/BCB_nonclone.csv', './BCB_nonclone_sample', './npy/')
    calc.get_distance()









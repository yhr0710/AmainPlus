import numpy as np
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


class DistanceCalculator:
    def __init__(self, ori, out, npy_path):
        self.ori = ori
        self.out = out
        self.npy_path = npy_path
        self.a = self.npy_path

    def listdir(self, path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.listdir(file_path, list_name)
            else:
                list_name.append(file_path)

    def get_distance(self, ori, out, npy_path, feature_number):
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
        existnpy = []
        # Recursively list all .npy files in the given directory and subdirectories
        self.listdir(npy_path, existnpy)
        j = 0

        exc = []
        reader = csv.reader(open(ori, 'r'))

        weightorder = []
        with open('weight1.txt', 'r') as f:
            lines = f.readlines()[:400]
            for line in lines:
                value = line.split()[1]
                weightorder.append(int(value))
        print(weightorder)
        print(len(weightorder))

        for r in reader:
            # Extract filenames from the first two columns of the CSV and strip the '.java' suffix
            f1 = r[0].split('.java')[0]
            f2 = r[1].split('.java')[0]
            # Construct the paths to the corresponding .npy files
            file1 = './npy/' + f1 + '.npy'
            file2 = './npy/' + f2 + '.npy'

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


        # print(len(exc[0]))
        # Write all computed distances to a new CSV file named based on the 'out' parameter
        with open(out + '_4_dis.csv', 'w', newline='') as csvfile0:
            writer = csv.writer(csvfile0)
            for row in exc:
                writer.writerow(row)


if __name__ == '__main__':
    calc = DistanceCalculator('./WT3T4.csv', 'WT3T4_clone','./npy/')
    calc.get_distance()









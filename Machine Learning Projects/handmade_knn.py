import numpy as np
import pandas as pd


class MyKNN:
    def __init__(self, path, k = 9, picture_data = False):
        '''
        Constructor for KNN object.
        Takes in training data file path and stores normalized pixel and label training data in Dataframes for later classifications.
        :param path: File path for MNIST training .csv
        :param k: K value for KNN classifier
        '''
        data = pd.read_csv(path)
        self.labels = data['label']
        self.data = data.drop('label', axis=1)
        if picture_data:
            self.data = self.data / 255  # Normalize training pixel data; min=0, max=255
        self.k = k

    def set_k(self, k):
        '''
        Sets k value for KNN classification.
        '''
        self.k = k

    def predict(self, path, picture_data = False):
        '''
        Predicts classes of a batch of test samples given in passed .csv.
        Outputs the accuracy of prediction.
        :param path: File path for MNIST testing .csv
        '''
        tests = pd.read_csv(path)
        real_labels = tests['label']
        test_data = tests.drop('label', axis=1)
        if picture_data:
            test_data = test_data / 255  # Normalize test pixel data; min=0, max=255

        # Apply classification to every instance in testing data, store predicted labels in predictions
        predictions = test_data.apply(self.get_prediction, axis=1)

        # Output accuracy for current predictions by counting number of correct predictions and dividing by number of all predictions
        print(f"k = {self.k}, Accuracy: {(predictions == real_labels).sum() / real_labels.size}")

    def get_prediction(self, test_instance):
        '''
        Classify label for single testing sample.
        :param test_instance: Pandas Series representing current test sample to predict digit class for
        :return: Predicted digit label
        '''
        # Calculate Euclidean distances (no sqrt) of test sample from every instance in training data
        dists = pd.DataFrame(((np.tile(test_instance.to_numpy(), (self.data.shape[0],1)) - self.data.values) ** 2).sum(axis=1), columns=['dist'])
        dists['label'] = self.labels

        # Return most frequent (the mode) label present in K smallest (closest) distances, return only first prediction if tie.
        return dists.nsmallest(self.k,'dist')['label'].mode().iloc[0]


# Create KNN object with training data
# MNIST training data must be in same director as this .py
path = input('Enter training data file path: ')
knn = MyKNN(path)

# Run automated KNN predictions for k = 1-9
print("Running KNN for k = 1,3,5,7,9")
for i in range(1,11,2):
    knn.set_k(i)
    knn.predict('MNIST_test.csv')

# User supplied KNN predictions
while True:
    k = input('Enter k: ')
    if k.lower() == 'q':
        break

    try:
        k = int(k)
    except:
        continue

    if k < 1:
        continue

    knn.set_k(k)
    test_path = input('Enter testing data file path:')
    knn.predict(test_path)

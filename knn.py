import math
import numpy as np


class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k
        self.data = []
        self.labels = []

    def fit(self, X_train, y_train):
        self.data = X_train
        self.labels = y_train

    def predict(self, X_test):
        classify = []

        for test_data in X_test:
            distance_label = []

            for i in range(len(self.data)):
                dist = self.euclideanDistance(test_data, self.data[i])
                distance_label.append((dist, self.labels[i]))
            distance_label.sort(key=lambda x: x[0])

            score = {}
            for i in range(self.k):
                key = distance_label[i][1]
                if key in score.keys():
                    score[key] += 1
                else:
                    score[key] = 1

            point_class = max(score, key=lambda k: score[k])
            classify.append(point_class)

        return classify

    def score(self, y_pred_test, y_test):
        num = np.sum(y_pred_test == y_test)
        den = len(y_test)
        return num / den

    def euclideanDistance(self, instance1, instance2):
        distance = 0.0
        for i in range(len(instance1)):
            distance += pow(instance1[i] - instance2[i], 2)
        return math.sqrt(distance)



import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution
import random

# --- Task 1 ---#
class MyClassifier:
    def __init__(self, K, iteration, lr):
        self.K = K  # number of classes
        self.iteration = iteration
        self.lr = lr
        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.w = None
        # self.b = None
        self.r = self.K
        self.c = None
        self.w = None
        self.b = np.zeros(self.r)

    def train(self, trainX, trainY):
        ''' Task 1-2
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm
        '''
        self.labels = list(map(int, set(trainY)))

        self.c = trainX.shape[1]
        self.w = np.zeros((self.r, self.c))

        for k in range(self.r):
            # transform trainY
            new_trainY = np.where(trainY == self.labels[k], 1, -1)
            for _ in range(self.iteration):
                for x, y in zip(trainX, new_trainY):
                    # predict the labels and calculate the error
                    labels = np.where(np.dot(x, self.w[k]) + self.b[k] > 0, 1, -1)
                    error = y - labels

                    # update the parameters
                    self.w[k] += self.lr * error * x
                    self.b[k] += self.lr * error
            print('class {} finished'.format(k))

    def predict(self, testX):
        ''' Task 1-2
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''

        predY = []
        for x in testX:
            predY.append(self.labels[np.argmax(self.w @ x + self.b)])
        # Return the predicted class labels of the input data (testX)
        return predY

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples: 
        # self.cluster_centers_ = None
        self.centers = []
        
    
    def train(self, trainX):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that describe the identified clusters
        '''
        # assign initial centers
        for i in range(self.K):
            self.centers.append(trainX[i])

        # initialize labels
        self.labels = [-1] * len(trainX)

        while True:
            # assign labels based on L1 distance to the center points
            new_labels = []
            for data_point in trainX:
                min_dis = None
                label = None
                for idx, c in enumerate(self.centers):
                    dis = np.linalg.norm((data_point - c), ord=2)
                    if min_dis is None or dis < min_dis:
                        label = idx
                        min_dis = dis
                new_labels.append(label)

            # check if the change is great enough
            change = 0
            for old_label, new_label in zip(self.labels, new_labels):
                change += 1 if old_label != new_label else 0

            # break if the change is small, otherwise update self.labels and self.centers
            if change < 0.02 * len(trainX):
                break
            else:
                self.labels = new_labels
                for i in range(self.K):
                    sum = np.zeros(len(trainX[0]))
                    cnt = 0
                    for idx, label in enumerate(self.labels):
                        if label == i:
                            sum += trainX[idx]
                            cnt += 1
                    self.centers[i] = sum / cnt

        # Update and return the cluster labels of the training data (trainX)
        self.labels = np.array(self.labels)
        return self.labels
    
    
    def infer_data_labels(self, testX): #predict the label
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''
        pred_labels = []
        for x in testX:
            dis = []
            for center in self.centers:
                dis.append(np.linalg.norm((x - center), ord=2))
            pred_labels.append(np.argmin(dis))
        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_data_labels(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy

    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i, 1, 0)
            num = np.bincount(true_labels[index == 1]).argmax()
            label_reference[i] = num

        return label_reference

    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio, k):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm
        self.k = k  # number of classes


    def select(self, trainX):
        ''' Task 3-2'''
        self.r = trainX.shape[0]
        cnt = int(self.r * self.ratio)

        # obtain cluster centers using MyClustering()
        model = MyClustering(self.k)
        model.train(trainX)
        centers = model.centers
        labels = model.labels

        dist_list = []
        data_to_label = []

        # Calculate the distance from each data point to its corresponding cluster center
        for idx, label in enumerate(labels):
            dist = np.linalg.norm((trainX[idx] - centers[label]), ord=2)
            dist_list.append((dist, idx))

        # Sort the list according to the distance
        dist_list.sort(key=lambda x:x[0])

        # Calculate the number of outliers (Do not use unless ratio >= 0.9)
        outlier = int(self.r * 0.05)
        if self.ratio >= 0.9:
            outlier = 0

        # Obtain (ratio/2)% the nearest points and (ratio/2)% furthest points
        for i in range(outlier, outlier + cnt//2):
            data_to_label.append(dist_list[i][1])
            data_to_label.append(dist_list[-i - 1][1])

        if cnt % 2:
            data_to_label.append(dist_list[cnt//2 + 1][1])

        # Return an index list that specifies which data points to label
        return data_to_label


    
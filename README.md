## Machine Learning Algorithms
This project is to learn how to implement basic machine learning algorithms from scratch.

### K-Nearest Neighbours
The first project implemented is KNN. 
The wine data set was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/
datasets/wine).

#### TODO 
- [ ] Create a report.pdf to show
    - [x] Predicted class labels of each instance in the test set using k=1 with min-max normalization for each feature
    - [x] The classification accuracy for k=1 on the test set with min-max normalization for each feature
    - [x] Classification accuracy for k=3, compare against k=1
    - [x] Discuss advantages/disadvantages of kNN method.
    - [x] Discuss the impact of increasing and decreasing k (too large/small)
    - [x] Describe applying k-fold cross validation for the dataset where k=5 (number of folds)
        - [x] - State the major steps 


#### KNN Report Info
(a)

Predicted class labels of each instance with k=1.

[3, 3, 3, 1, 1, 1, 1, 2, 1, 2, 2, 3, 3, 3, 1, 2, 3, 3, 1, 1, 3, 2, 2, 3, 2, 3, 2, 3, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 3, 1, 2, 1, 3, 2, 2, 1, 3, 1, 1, 3, 3, 1, 1, 3, 1, 3, 3, 1, 2, 3, 2, 3, 3, 1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 3, 1, 1, 2, 2, 3, 1, 2, 1, 1, 2, 1]

Classification accuracy of k=1: 0.9438202247191011.

(b)

Classification accuracy for k=3: 0.9550561797752809. k=3 performed slightly better than k=1, in terms of accuracy.

(c)

Main advantages of k Nearest Neighbours:
* It is a simple algorithm to implement.
* There is no training time.

Main disadvantages of k Nearest Neighbours:
* High cost for testing time, as it has to compare every training observation against every test observation.
* KNN is very susceptible to the curse of dimensionality due to the distance metric. As the number of features grow, the distance between observations grows too. 
With every additional feature, we need to do n more comparisons, where n is the number of observations. 

Impact of increasing/decreasing k:
* The higher the k-value, the larger the boundary for classification. This is a problem for uneven datasets especially. 
If we were to increase k to the total size of the dataset, we would see every single node be predicted as the node that appears the most in the training data.
* The lower the k-value, the more susceptible the prediction is to outliers. 

(d)

To apply k-fold cross validation with k_folds=5, we need to:
* Join training and testing data, then randomly shuffle it.
* Split our data into 5 equal parts.
* Then use 4 parts to train and 1 part to test. Fit the model on the 4 training parts, test it and then evaluate it.
* Keep the evaluation score (MSE), but do not keep the model.
* Continue cycling out the test part for each of the training parts and use the old test parts as training parts. 
* Repeat this k (5) times total so each part is used to train once and test k-1 (4) times.
* Calculate the average MSE, this will be our indication of how well the model will perform on new data.


### Decision Tree Report Info



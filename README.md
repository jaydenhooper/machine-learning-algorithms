## Machine Learning Algorithms
This project is to learn how to implement basic machine learning algorithms from scratch.

### K-Nearest Neighbours

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


### Decision Tree Report

(a) The accuracy of my decision tree on the hepatitis data set is 0.7411 (4dp). 
Considering the baseline accuracy is 0.8, this is a poor result.

HISTOLOGY = True: 
	SGOT = True: 
		BILIRUBIN = True: 
			VARICES = True: 
				ASCITES = True: 
					Class die, prob=1
				ASCITES = False: 
					Class live, prob=1
			VARICES = False: 
				ASCITES = True: 
					SPIDERS = True: 
						Class live, prob=0.8
					SPIDERS = False: 
						SPLEENPALPABLE = True: 
							Class live, prob=1
						SPLEENPALPABLE = False: 
							Class die, prob=1
				ASCITES = False: 
					Class live, prob=0.8
		BILIRUBIN = False: 
			Class live, prob=1
	SGOT = False: 
		BILIRUBIN = True: 
			VARICES = True: 
				ASCITES = True: 
					Class live, prob=1
				ASCITES = False: 
					Class die, prob=1
			VARICES = False: 
				ASCITES = True: 
					SPIDERS = True: 
						Class live, prob=0.8
					SPIDERS = False: 
						SPLEENPALPABLE = True: 
							Class live, prob=1
						SPLEENPALPABLE = False: 
							FIRMLIVER = True: 
								BIGLIVER = True: 
									ANOREXIA = True: 
										Class live, prob=1
									ANOREXIA = False: 
										Class die, prob=1
								BIGLIVER = False: 
									Class live, prob=0.8
							FIRMLIVER = False: 
								Class live, prob=0.8
				ASCITES = False: 
					Class live, prob=0.8
		BILIRUBIN = False: 
			VARICES = True: 
				ASCITES = True: 
					SPIDERS = True: 
						Class live, prob=1
					SPIDERS = False: 
						Class die, prob=1
				ASCITES = False: 
					Class live, prob=0.8
			VARICES = False: 
				Class live, prob=0.8
HISTOLOGY = False: 
	Class live, prob=1


(b) (i)
There are a couple different criteria that can be used to decide which leaves can be pruned:
* The number of instances in a leaf. If the number of instances is less than a certain threshold, then the leaf can be pruned. This improves resistance to outliers in the training data, making it more generalizable to unseen data.
* The depth of a leaf. By asking too many questions, we can overfit the data. Hence, if the depth of the leaf is too large, we can prune it.

(b) (ii)
Pruning reduces the accuracy on the training data because we are re-characterizing the training data. This re-characterization is done to capture the general trends in the data, attempting to ignore the noise.

(b) (iii)
We expect the accuracy of the test set to increase after pruning. The whole purpose of pruning is to generalize better to unseen data. 


### Perceptron Report

(a) I would argue the perceptron found a correct set of weights. The accuracy of the perceptron when training and testing on the same data is 0.8476 (4dp). The accuracy of the perceptron on the after splitting the data is 0.8476 (4dp) for a 70%/30% split (training/testing). These are decent result, as it is better than the baseline accuracy of 0.8. It did not change between different runs.

(b) The perceptron is trying to minimize the error in the training data so it will be heavily biased toward this data. Hence, using the same data to measure the performance of the perceptron will not give us a good indication of how well the perceptron will perform on unseen data.


Output Requested:

For testing on the data the perceptron was trained on:
Iterations until convergence 198
Accuracy: 0.8857142857142857
Weights:
 [ 57.        0.        8.30363   2.66221  10.66447   7.97673   0.37107
  12.36111  15.27071   4.61155  -8.46069  -3.8096   -3.43787  -0.17199
  11.86249  -9.53405   2.70213   7.9501  -15.27281   2.18966  -0.50453
 -11.53109  10.82663   8.94307   5.28427   3.6466  -16.03456   0.8899
   4.54515   6.3531    6.89108   2.30255   1.1352  -12.1055 ]

For train/test split data:
Iterations until convergence  161
Accuracy: 0.8476190476190476
Weights:
 [ 5.600000e+01  0.000000e+00  9.548790e+00 -4.270000e-01  6.355950e+00
 -1.761210e+00  4.655940e+00  1.241193e+01  7.564210e+00  6.297910e+00
 -4.762670e+00  8.344550e+00  3.460230e+00  6.158240e+00  5.105000e-02
 -1.794331e+01 -1.214350e+00  1.006977e+01 -1.252454e+01  8.160400e-01
  1.326059e+01 -2.329778e+01  1.116332e+01 -8.378530e+00  1.283087e+01
  1.682397e+01 -3.029886e+01  1.181260e+00  1.358000e+00  8.817420e+00
  1.663547e+01  9.731200e-01 -3.044800e-01 -8.508990e+00]
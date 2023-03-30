# %% [markdown]
# ## Perceptron

# %% [markdown]
# ### Import Data
# 

# %%
import pandas as pd
import numpy as np
import sys

# %%
train_fname = sys.argv[1]
iono = pd.read_csv(train_fname, delimiter=" ")
iono = pd.read_csv(train_fname , delimiter = " ",
                   converters = {34: lambda x: 1 if x == 'g' else 0})

iono_X = iono.drop("class", axis=1)
iono_y = iono["class"]

# %% [markdown]
# ### Split Data

# %%
iono_train = iono.sample(frac=0.7, random_state=0)
iono_test = iono.drop(iono_train.index)

iono_train_X = iono_train.drop("class", axis=1)
iono_train_y = iono_train["class"]
iono_test_X = iono_test.drop("class", axis=1)
iono_test_y = iono_test["class"]

# %% [markdown]
# ### Perceptron Class

# %%
class Perceptron:
    def __init__(self):
        import numpy as np
        self.weights = None
        self.bias = None
    
    def train(self, X, y):
        """Train the perceptron on the given data."""
        n_observations, n_attributes = X.shape
        # initialize parameters
        self.weights = np.zeros(n_attributes)
        self.bias = 0
        # update weights for as many iterations specified
        counter, total_count, max_improvement, current_improvement = 0, 0, 0, 0
        while (counter < 100 or max_improvement == 1):
            total_count += 1
            for i, row in enumerate(X):
                z = np.dot(row, self.weights) + self.bias
                y_pred = self.__activation(z)
                update = (y[i] - y_pred)
                self.weights += update * row
                self.bias += update
            current_improvement = self.evaluate(X, y)
            if(max_improvement < current_improvement):
                max_improvement = current_improvement
                counter = 0
            counter += 1
        return total_count

    def predict(self, X):
        """Predict the class labels for the given data."""
        if self.weights is None:
            raise Exception("Perceptron has not been trained. Please call .train() first.")
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.__activation(z)
        return y_pred
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)

    def __activation(self, z):
        """Activation function, given the bias is a weight."""
        return np.where(z > 0, 1, 0)

p1 = Perceptron()
p1_num_iters = p1.train(iono_X.to_numpy(), iono_y.to_numpy())
print("\nFor testing on the data the perceptron was trained on:")
print("Iterations until convergence:", p1_num_iters)
print("Accuracy:", p1.evaluate(iono_test_X, iono_test_y))
print("Weights:\n", p1.weights)

# %%
p2 = Perceptron()
num_iters = p2.train(iono_train_X.to_numpy(), iono_train_y.to_numpy())
print("\nFor train/test split data:")
print("Iterations until convergence:", num_iters)
print("Accuracy:", p2.evaluate(iono_test_X, iono_test_y))
print("Weights:\n", p2.weights)




# %% [markdown]
# # Decision Tree

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# %% [markdown]
# ## Loading Data

# %%
train_fname = sys.argv[1]
test_fname = sys.argv[2]

hep_train = pd.read_csv(train_fname, delimiter=" ")
hep_test = pd.read_csv(test_fname, delimiter=" ")

# %%
hep_train_X = hep_train.drop("Class", axis = 1)
hep_train_y = hep_train["Class"]
hep_test_X = hep_test.drop("Class", axis = 1)
hep_test_y = hep_test["Class"]


# %% [markdown]
# ## Decision Tree Algorithm

# %%
class DecisionTree:
    def __init__(self):
        import numpy as np
        import pandas as pd
        from scipy import stats
        
    def train(self, train_X: pd.DataFrame, train_y: pd.Series):
        self.train_y = train_y
        self.baseline = self.__most_common_class(train_y)
        self.class_name = train_y.name
        attributes = train_X.columns
        # concatenating x & y will make it easier for the recursive structure of the algorithm
        instances = pd.concat([train_y.to_frame(), train_X], axis = 1) # note y = column 0
        self.node = self.__build_tree(instances, attributes)
        return self.node
    
    def test(self, test_X: pd.DataFrame) -> pd.Series:
        """Tests the model on the test set and returns the predictions."""
        if(not self.node):
            raise Exception("DecisionTree not trained yet. Please call .train() first.")
        predicted_nodes = []
        for i in range(len(test_X)):
            predicted_nodes.append(self.__test_helper(test_X.iloc[i], self.node))
        return pd.Series(predicted_nodes)

    def __test_helper(self, test_X_row: pd.Series, node):
        """Helper function for the test function. Recursively traverses the tree to find the leaf node."""
        if(isinstance(node, LeafNode)):
            return node.class_name
        split_class = node.class_name
        split_value = test_X_row[split_class] # type: ignore
        if(split_value):
            return self.__test_helper(test_X_row, node.true_branch)
        return self.__test_helper(test_X_row, node.false_branch)
    
    def evaluate(self, pred: pd.Series, test_y: pd.Series):
        """Calculates the accuracy of the predictions."""
        return np.mean(pred == test_y)
    
    def __build_tree(self, instances: pd.DataFrame, attributes: pd.Index):
        """Builds an optimal structured decision tree with decision nodes and leaf nodes."""        
        # to determine the root node, we find the first question with the optimal value
        if(instances.empty):
            # returns leaf node that contains the name and probability of the most probable class
            return LeafNode(self.baseline, np.max(self.__frequency(self.train_y))) 
        unique_class_values = instances[instances.columns[0]].unique()
        if(len(unique_class_values) == 1):  # check for pure node
            # returns leaf node that contains the name of the unique class with prob 1
            return LeafNode(unique_class_values[0], 1)  
        if(attributes.empty):
            # returns leaf node that contains the name and probability of the majority class of the instances
            return LeafNode(self.__most_common_class(instances[instances.columns[0]]),  # type: ignore
                                    self.__frequency(instances[instances.columns[0]])) # type: ignore
        lowest_impurity = 1
        best_att = attributes[1]
        best_insts_true = instances
        best_insts_false = instances
        for attribute in attributes:
            # separate instances into two sets
            insts_true =  instances[instances[attribute]]
            insts_false = instances[instances[attribute] == False]
            # calculate purity of each set
            gini_true  = self.__gini_impurity(insts_true [self.class_name]) # type:ignore
            gini_false = self.__gini_impurity(insts_false[self.class_name]) # type:ignore
            # weighted average purity
            weighted_avg_impurity = ((len(insts_true)  / len(instances)) * gini_true + 
                                   (len(insts_false) / len(instances)) * gini_false)
            if(weighted_avg_impurity < lowest_impurity):
                best_att = attribute
                best_insts_true  = insts_true
                best_insts_false = insts_false
        left  = self.__build_tree(best_insts_true,  attributes.drop(best_att))
        right = self.__build_tree(best_insts_false, attributes.drop(best_att))
        node = DecisionNode(best_att, left, right)
        return node
    
    def __gini_impurity(self, class_label: pd.Series) -> float:
        """Calculates the gini impurity for the given labels."""
        freq = self.__frequency(class_label)
        return 1 - np.sum([f ** 2 for f in freq]) # type: ignore
    
    def __frequency(self, class_label: pd.Series) -> list[int]:
        """Returns a list of the frequencies of each label."""
        return [np.mean(value == class_label) for value in class_label.unique()] # type: ignore
    
    def __most_common_class(self, class_label: pd.Series):
        freq = self.__frequency(class_label)
        unique = class_label.unique()
        return unique[np.argmax(freq)]
    
class DecisionNode:
    """Decision nodes hold the optimal question at this level and two child nodes."""
    def __init__(self, class_name, true_branch, false_branch):
        self.class_name = class_name
        self.true_branch = true_branch
        self.false_branch = false_branch
        
    def report(self, indent: str):
        print(f"{indent}{self.class_name} = True: ")
        self.true_branch.report(indent+"\t")
        print(f"{indent}{self.class_name} = False: ")
        self.false_branch.report(indent+"\t")


class LeafNode:
    """Leaf nodes return the predicted class."""
    def __init__(self, class_name, probability):
        self.class_name = class_name
        self.probability = probability
    
    def report(self, indent: str):
        if(self.probability==0): 
            print(f"{indent}Unknown%n")
        else:
            print(f"{indent}Class {self.class_name}, prob={np.round(self.probability, 2)}")

# %% [markdown]
# ## Applying the Algorithm

# %%
dt_hep = DecisionTree()
root = dt_hep.train(hep_train_X, hep_train_y)
root.report("")

# %%
pred = dt_hep.test(hep_test_X)
accuracy = dt_hep.evaluate(hep_test_y, pred)
print("Accuracy:", accuracy)


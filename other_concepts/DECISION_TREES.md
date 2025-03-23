# Decision Trees

Decision Trees are a supervised learning algorithm used for both classification and regression tasks. They work by recursively splitting the data into subsets based on feature values, creating a tree-like structure. Decision Trees are popular due to their simplicity, interpretability, explainability, and ability to handle both numerical and categorical data.

---

## Table of Contents
- [Explainability](#explainability)
  - [How Decision Trees Work](#how-decision-trees-work)
  - [Feature Importance](#feature-importance)
  - [Advantages of Explainability](#advantages-of-explainability)
- [Gini Index and Information Splits](#gini-index-and-information-splits)
  - [Gini Index](#gini-index)
  - [Information Gain](#information-gain)
    - [Shannon Entropy](#shannon-entropy)
  - [Example of Splitting](#example-of-splitting)
- [Ensemble Models](#ensemble-models)
  - [Random Forest](#random-forest)
  - [Gradient Boosted Decision Trees (GBDT)](#gradient-boosted-decision-trees-gbdt)
  - [Bagging vs Boosting](#bagging-vs-boosting)
- [Conclusion](#conclusion)

---

## Explainability

### How Decision Trees Work
Decision Trees split the data into subsets based on feature values. Each split is chosen to maximize the separation of the target variable. The tree consists of:
- **Root Node**: The starting point of the tree, representing the entire dataset
- **Internal Nodes**: Represent decisions based on feature values
- **Leaf Nodes**: Represent the final output (class label or regression value)

For example, in a classification task, a decision tree might ask:
1. Is the feature `age > 30`?
2. If yes, is `income > 50k`?
3. If no, is `education = college`?

### Feature Importance
Decision Trees provide a natural way to measure **feature importance**. Features that are used closer to the root of the tree (i.e., those that result in the largest information gain or Gini reduction) are considered more important

Feature importance can be calculated as:
- The total reduction in impurity (e.g., Gini index or entropy) contributed by a feature across all splits in the tree.

### Advantages of Explainability
1. **Human Interpretability**:
   - Decision Trees are easy to visualize and understand, making them suitable for domains where interpretability is critical (e.g., healthcare, finance).
2. **Transparency**:
   - The decision-making process is transparent, as the splits and thresholds are explicitly defined.
3. **Debugging**:
   - Misclassifications can be traced back to specific splits, helping to debug the model.

---

## Gini Index and Information Splits
One of the best ArXiv papers I've ever read was on [Understanding Shannon's Entropy Metric for Information](https://arxiv.org/pdf/1405.2061) which is a very useful paper for understanding how we can obtain information from different signals

### Gini Index
The **Gini Index** is a metric used to measure the impurity of a node in a classification task. It quantifies how often a randomly chosen element from the dataset would be incorrectly classified if it were randomly labeled according to the distribution of labels in the node.

The formula for the Gini Index is:
$Gini = 1 - \sum_{i=1}^{C} p_i^2$

Where:
- $C$ is the number of classes.
- $p_i$ is the proportion of samples belonging to class $i$ in the node.

A Gini Index of 0 indicates a pure node (all samples belong to one class), while a higher Gini Index indicates more impurity.

### Information Gain
**Information Gain** measures the reduction in entropy after a split. It is used to decide which feature to split on at each step in the tree.

The formula for Information Gain is:
$IG = Entropy_{parent} - \sum_{k=1}^{n} \frac{|D_k|}{|D|} \cdot Entropy_k$
Where:
- $Shannon Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$
- $D_k$ is the subset of data after the split.

#### Shannon Entropy
    - $log_2(p_i)$ is equivalent to "number of bits needed to represent choices of $p$" - meaning if there are 4 total values, $log_2(4) = 2$ total bits are needed to represent all of the potential outcomes
        - If a specific $p$ is $p(x) = 2/4$, then the number of bits $ \text{bits} = log_2(4) - log_2(2) = 1$ 1 total bit
    - So to find the total amount of storage we need for all of our potential values, we need to sum up the total storage bits multiplied by their proportion in our output distribution, if the $p$ that required 1 bit occured 99% of the time, we could almost store the enitre output in 1 bit!
        - Therefore, to find out the total number of bits we need to store all of our information, we use $sum_{i=1}^{C} p_i \log_2(p_i)$ which is our ***formula for entropy!***

### Example of Splitting
Consider a dataset with the following labels:
- Parent Node: [Yes, Yes, No, No, Yes]
- Split 1: [Yes, Yes] and [No, No, Yes]

#### Gini Index Calculation:
- Parent Node Gini:
$Gini = 1 - (p_{Yes}^2 + p_{No}^2) = 1 - \left(\left(\frac{3}{5}\right)^2 + \left(\frac{2}{5}\right)^2\right) = 0.48$
- After Split:
  - Left Node: [Yes, Yes], $Gini = 1 - (1^2 + 0^2) = 0$
  - Right Node: [No, No, Yes], $Gini = 1 - \left(\left(\frac{1}{3}\right)^2 + \left(\frac{2}{3}\right)^2\right) = 0.44$

The split with the largest reduction in Gini Index is chosen.

---

## Ensemble Models

### Random Forest
**Random Forest** is an ensemble method that combines multiple decision trees to improve performance and reduce overfitting. It works by:
1. Training multiple decision trees on different subsets of the data (using bootstrapping)
2. Using a random subset of features for each split in a tree
3. Aggregating the predictions of all trees (e.g., majority vote for classification, average for regression)

#### Advantages:
- Reduces overfitting compared to a single decision tree
- Handles high-dimensional data well

### Gradient Boosted Decision Trees (GBDT)
**GBDT** is another ensemble method that builds trees sequentially, where each tree corrects the errors of the previous one. It minimizes a loss function (e.g., mean squared error for regression, log loss for classification) using gradient descent

#### Key Features:
- Highly accurate and widely used in competitions (e.g., XGBoost, LightGBM, CatBoost)
- Can handle non-linear relationships effectively

### Bagging vs Boosting
- **Bagging** (e.g., Random Forest):
  - Trains multiple models independently on different subsets of data
  - Reduces variance
- **Boosting** (e.g., GBDT):
  - Trains models sequentially, with each model correcting the errors of the previous one
  - Reduces bias

---

## Conclusion
Decision Trees are a versatile and interpretable machine learning algorithm. They are particularly powerful when combined into ensembles like Random Forests and Gradient Boosted Decision Trees. By understanding concepts like the Gini Index and Information Gain, we can better interpret how decision trees make splits and why they are effective in various tasks.
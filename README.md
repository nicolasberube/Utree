# Utree

## What is it for

This class does segmentation based of the efficiency of a specific binary variable to affect another one.

Originally, this code was developed to find the areas where a certain intervention (bin_data = True or False depending on if the intervention was performed) on the reduction of injuries (y_data = difference in yearly injuries before and after).

This code will select subsection of the dataset based on other variables like time, size, location (all documented in split_data) to pinpoint in which subsection the intervention is the most (or least) efficient.

For example, let's consider a case where an intervention normally lowers the yearly injury rate by 3. However, if each site's local region is documented in split_data, the tree will measure if this effect is global, or if it's localized. Indeed, the inspection might lower the average yearly injury count by 5 in region A, and by 1 in region B. In this case, it will find "region=A or B" as a delimiter in the decision tree.

The normal way to apply a decision tree on this problem would be to calculate the effect of bin_data on y_data (in our previous case, the reduction of 1 or 5 injuries in each region), and apply a standard decision tree algorithm on this data.
However, such an approach does not consider the **statistical strength** of the effect of bin_data on y_data. This algorithm is an attempt at correcting this.

The goal of the model is to figure out in which subset of our data certain types of actions are more efficient so we can concentrate on those. The decision tree algorithm was therefore chosen since it is fully interpretable and can be converted in a clear policy and guidelines for agents to easily use. Therefore, the model needs to have a real predictive power when applied to real-life cases, hence why the statistical strength of the prediction needed to be incorporated in the algorithm.

## How to use

You need to specify three datasets: y_data (1D), split_data (2D) and bin_data (1D bool).
You can also specify the flag max_tree_depth and/or value_threshold. The code has been written with numpy arrays as input, but you should be able to use pandas dataframe if you provide ```labels=list(split_data)```.

Once the utree class has been initialized, the tree itself can be computed with the class' ```compute_tree()``` function.

```
U = utree(y_data, split_data, bin_data)
U.compute_tree()
```

The tree is available as U.tree, and a row of split_data can be attributed a leaf with ```U.tree_to_leaf(x_row)```.

The leaves attributes are explained in the ```node()``` class.

## How does it work

This is just a standard decision tree algorithm, but instead of only predicting the y_data based on split_data (with regularization techniques), it also checks how different the distributions are as a measure of the statistical significance. Since the distributions tended to be zero-skewed in our test cases, a non-parametric test was chosen: the Mann-Whitney U-test (more on that later).

The impact of bin_data on y_data is measured by analyzing the distributions A = y_data|bin_data=True and B = y_data|bin_data=False.
Three factors are considered to measure the impact of the intervention (bin_data) on y_data:
- The difference in means between A and B is high
- The U-test value between A and B is low
- The amount of data points in A and B is high

The goal of our algorithm is to create a partition where the distributions A and B on each leaf will be such as to maximize those factors. Therefore, for each node, every possible cutoff is considered (for every value of every variable in split_data). Each cutoff separates the dataset in two, where split_data[i] <= cutoff_i and split_data[i] > cutoff_i. To calculate how good a cutoff is at maximizing our three factors, we consider the four following distributions:

- A1 = y_data|split_data[i] <= cutoff_i & bin_data=True
- B1 = y_data|split_data[i] <= cutoff_i & bin_data=False
- A2 = y_data|split_data[i] > cutoff_i & bin_data=True
- B2 = y_data|split_data[i] > cutoff_i & bin_data=False

The cutoff should partition the data where the effect of bin_data is different between leaves. We calculate it as effect = mean(A) - mean(B). Therefore, we want the cutoff to maximize the absolute value of

```
|effect(1) - effect(2)| = |[mean(A1) - mean(B1)] - [mean(A2) - mean(B2)]|
```

We also want the cutoff to make sure the effect on a distribution is statistically significant. For this, we compute the U-test between A and B for both distributions 1 and 2 created by the cutoff.

The U-test value U is normally between 0 and N_A * N_B/2, half the product of the number of data points in A and in B. We therefore normalize the U-test value to be between 0 and 1 with

```
r = 1 - 2*[U-test value]/(N_A*N_B)
```

Here, r is interpreted the proportions of all possible data point pairs between A and B that shows an advantage to one of the samples (normalized between 0 and 1 since no effect would mean exactly half of the pairs would show an advantage). Therefore, we want the cutoff to maximize the absolute value of r for both distributions 1 and 2.

Finally, we want to maximize the number of data points in each distributions to avoid overfitting. Since this algorithm was designed on a dataset where inspections were rare compared to the amount of available sites (i.e. the number of data points in A is very small), we want the cutoff to maximize the number of data points in the A distributions (the code actually takes the minimum of N_A and N_B in the case that N_B < N_A).

For each possible cutoff the "value" of the cutoff is computed as the product of all those quantities:

```
cutoff_value = |effect(1) - effect(2)| * r(1) * r(2) * min(N_A1, N_B1) * min(N_A2, N_B2)
```

The cutoff with the highest value is kept, and the tree branches from there. Then, the process repeats for all node until either a maximum tree depth is reached, of no cutoff has a value above a specified threshold.

Every cutoff value is actually computed in a brute force fashion, which is very inefficient compared to more modern algorithm. However, there is a flag (use_scipy) that, when turned off, the U-test value is computed manually instead of from a scipy library. This allows the reuse of a lot of data from one cutoff value calculation to another (in the same split_data column) to accelerate the whole process.

## To-do

### Better statistical test / more interpretable cutoff value

For now, the cutoff value is hard to interpret. Trials and errors implied that a good value_threshold is between 1000 and 5000 for my own dataset, but it is very hard to figure out what to put for good prediction accuracy.

The [P-value of the U-test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Normal_approximation_and_tie_correction) can be computed and used for this purpose.

### Less brute force

For now, each cutoff value is computed individually, even though there's a clever reuse of data calculation for the U-tree between cutoffs for large vectors. It is worth considering incorporating more modern approaches inspired by XGBoost into the algorithm to make it more efficient.

Also, better performance analysis regarding the use of scipy_flag are needed in the best_cutoff() function.

### Make it simpler

The U-test between the leaves (i.e. between distributions 1 and 2 of a cutoff) could be used as well, to consider the statistical efficiency of the decision tree itself, instead of the effect of bin_data.

Also, some situations do not have a bin_data, and are simply trying to predict y_data. It could be useful to reuse the classes function measuring statistical accuracy for a simpler case with only y_data and split_data.

### Model accuracy and predicting power

This algorithm was used to measure the efficiency of inspections to reduce injuries. Here are a few of the findings regarding the accuracy of this particular model and how it was measured. This code is not included (for now) in the current class.

The model was trained on data up to a certain date (2015). Then, data since 2015 was used to test the model.

For each of the Utree's leaf, we measure the real effect on y_data (averaged for all data points on the leaf) of the testing dataset, and the predicted effect on y_data my by the model. The accuracy is calculated with

```
leaf_acc = 1 - abs(model_effect - real_effect)/real_effect
```

This gives us the accuracy of how well we predicted the effect (reduction or augmentation) of injuries on this leaf. Note the accuracy can be negative if the effect is different sign (predicts reduction instead of augmentation) or is more than twice the size.

We then average (weighted by the number of data point on the leaf of the test data set) the accuracy for all leaves to get the overall model accuracy. This gives us our average accuracy of how well we predicted the injuries variation with our model

Note that for this particular case, the y_data was a three year moving average. Therefore, there was data pollution in our particular case since the moving average for 2015 to 2017 used data from before 2015, which was used to train the model.

Two other techniques were be implemented in our model to improve this accuracy
metric. The first one is optional.

  1. Smoothing effect over the tree

Because of the high variance of the injury data in our case, we had to articially reduce the variance of the model's prediction. We calculated the weighted average effect of all leaves and labeled it as the overall effect of the tree.

This overall effect is incorporated in the leaves where each leaf's effect is approximately 80% overall effect + 20% individual leaf effect.

The 80/20 split is actually the ratio of training data (before 2015) and testing data
(2015 and after).

2. Kill switch

The leaf accuracy metric was calculated on a proportion of the trained dataset (in our case, data from 2012 to 2015). Yes, this is a terrible metric since it means that the accuracy is calculated on a data set that served to train the model, and should be artificially high for that reason. However, certain leaves still had a negative accuracy, meaning the variance is very high. For each leaf where this was the case, the model will just predict an effect on y_data of zero instead (transforming the negative accuracy into a zero accuracy instead), and effectively "kill the leaf" for the real predictions on the test (or validation) dataset.

## Dependancies

```
numpy = 1.18.1
scipy.stats = 1.4.1
```

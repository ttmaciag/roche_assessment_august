# Titanic Assessment Project

This repository contains multiple tasks on the Titanic dataset. To see the instructions for each task got to [this document](docs/TASKS.md).


## Branch-specific comments:
There are two options for running models in this branch:
1. To run and test a single model use 'python src/pipeline.py. Inside this script you can choose the model.
2. Compare multiple model using 'python src/pipline_multi_model.py'.


### Model comparision:
Two additional models were implemented and compared with the existing RandomForest. 
Since the classes in the datasets are not perfectly balanced, Recall, Precision, and F1 were addionally calculated to get more insights on the performance of the models. Each model was train and evaluated 10 times in order to minimize the effect of random initializations. The averaged results are as follows:

| Model          | Accuracy | Precision | Recall |   F1  |
|----------------|----------|-----------|--------|-------|
| RandomForest   | 0.791    | 0.819     | 0.672  | 0.738 |
| GBDecisionTree | 0.800    | 0.861     | 0.649  | 0.739 |
| SVC            | 0.811    | 0.827     | 0.721  | 0.769 |

It should be noted, that it would be very difficult to obtain a perfect score at this task. As one can imagine, there were a lot of random events during the Titanic accident that are not captured in the dataset and which highly influenced the chance of survival of the passangers.

All models where used in their default hyper-parameter settings, due to time constraints. Just the number of estimators in the Random Forest and GBDT was set to 50, to make them more comparible.

### Explanation and Discussion:
*Gradient Boosting Decision Tree (GBDT)* was selected as the default Random Forest performed quite well and GBDT can be considered a more intricate approach to optimizing ensambles of Decision Trees. In general, GBDT obtained very similar scores the Random Forest. This might mean that the DecisionTree-based models struggle to surpass a certain Accuracy and F1 level due to their conceptual specifics, such as their tendecy to overfit (low bias, high variance).

*Support Vector Classifier (SVC)* was used as it is a relatively simple (linear) classifier that can deal well with classes which are not easily separable (and it is sensible to think that the Survival classes are overlapping). The SVC is a soft-margin classifier, so it is less susceptible to outliers. The results show that the SVC was the best classifier in general which is interesting, because the DecisionTree-based algorithms are in theory more flexible (non-linear). 

Basing on these results, one can argue, that on the Titanic dataset, it is important to have a model that has a higher bias, which is an advantege of the SVC. To proove this in the future,  we could see how a single Decision Tree preforms, as this ML technique is known for its low bias and high variance. On the other hand, we could try achieving even better scores than the SVC by using a Support Vector Machine with a non-linear kernel, such as a polynomial or RBF kernels.
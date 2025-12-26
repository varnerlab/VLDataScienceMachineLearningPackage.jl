# Binary Classification
We've implemented some basic functionality for binary classification tasks, including a simple perceptron model and logistic regression. This model can be trained on labeled data and used to classify new instances.

```@docs
VLDataScienceMachineLearningPackage.learn
VLDataScienceMachineLearningPackage.classify
VLDataScienceMachineLearningPackage.confusion
```

### K-Nearest Neighbors Classifier
In addition to the perceptron and logistic regression models, we've also included a K-Nearest Neighbors (KNN) classifier. This model classifies instances based on the majority class of their nearest neighbors in the feature space.

```@docs
VLDataScienceMachineLearningPackage.MyKNNClassificationModel
```

You can call the `classify` function with a test data vector and a MyKNNClassificationModel instance to get the predicted class label.
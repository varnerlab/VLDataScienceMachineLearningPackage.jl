# Bandit Algorithms
This section covers the implementation of several multi-armed bandit algorithms in the `VLDataScienceMachineLearningPackage.jl` package. 

```@docs
VLDataScienceMachineLearningPackage.MyExploreFirstAlgorithmModel
VLDataScienceMachineLearningPackage.MyEpsilonGreedyAlgorithmModel
VLDataScienceMachineLearningPackage.MyUCB1AlgorithmModel
VLDataScienceMachineLearningPackage.solve(model::AbstractBanditAlgorithmModel; T::Int = 0, world::Function = _null)
VLDataScienceMachineLearningPackage.regret
```

## Binary arms
Suppose we wanted to look at a bandit problem where each arm is represented by a binary vector of length K. This means that there are a total of 2^K possible arms, each corresponding to a unique combination of the K binary features. Further, the reward structure is computed in a user specified world function that maps each binary vector to a reward, along with a context model.

We've built in support for this type of bandit problem in the package. Specifically, we have defined a new abstract type `AbstractBanditAlgorithmContextModel` that can be used to represent the context of a bandit problem with binary arms. We have a concrete implementation of this type called `MyConsumerChoiceBanditContextModel`, which allows users to specify the average rewards for each possible combination of binary features.

```@docs
VLDataScienceMachineLearningPackage.MyBinaryVectorArmsEpsilonGreedyAlgorithmModel
VLDataScienceMachineLearningPackage.MyConsumerChoiceBanditContextModel
VLDataScienceMachineLearningPackage.solve(model::MyBinaryVectorArmsEpsilonGreedyAlgorithmModel, context::AbstractBanditProblemContextModel; T::Int = 0, world::Function = _null)
```
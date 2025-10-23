# Bandit Algorithms
This section covers the implementation of several multi-armed bandit algorithms in the `VLDataScienceMachineLearningPackage.jl` package. 

```@docs
VLDataScienceMachineLearningPackage.MyExploreFirstAlgorithmModel
VLDataScienceMachineLearningPackage.MyEpsilonGreedyAlgorithmModel
VLDataScienceMachineLearningPackage.MyUCB1AlgorithmModel
VLDataScienceMachineLearningPackage.solve(model::AbstractBanditAlgorithmModel; T::Int = 0, world::Function = _null)
VLDataScienceMachineLearningPackage.regret
```
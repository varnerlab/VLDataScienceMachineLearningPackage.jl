# Markov Decision Processes (MDPs)
We've developed some codes to work with Markov Decision Processes (MDPs). MDPs are mathematical frameworks used for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. They are widely used in various fields, including robotics, economics, and artificial intelligence, particularly in reinforcement learning.

## Types
We have defined the following abstract and concrete types to represent MDPs and related concepts:

```@docs
VLDataScienceMachineLearningPackage.MyMDPProblemModel
VLDataScienceMachineLearningPackage.MyRectangularGridWorldModel
VLDataScienceMachineLearningPackage.MyValueIterationModel
VLDataScienceMachineLearningPackage.MyValueFunctionPolicy
```

We construct the `MyMDProblemModel` type, and the `MyRectangularGridWorldModel` type to represent the environment in which the MDP operates using custom build methods:

```@docs
VLDataScienceMachineLearningPackage.build(model::Type{MyMDPProblemModel}, data::NamedTuple)VLDataScienceMachineLearningPackage.build(modeltype::Type{MyRectangularGridWorldModel}, data::NamedTuple)
```


The `MyValueIterationModel` type is used to define the parameters for the Value Iteration algorithm, and the `MyValueFunctionPolicy` type is used to store the results of the Value Iteration algorithm, including the optimal value function and policy.

These types allow us to create and manipulate MDPs, define the environment, and implement algorithms for solving MDPs.

## Value Iteration Algorithm
We have implemented the Value Iteration algorithm, which is a dynamic programming algorithm used to compute the optimal policy and value function for an MDP. The algorithm iteratively updates the value of each state based on the expected rewards and the values of successor states.

```@docs
VLDataScienceMachineLearningPackage.solve(model::MyValueIterationModel, problem::MyMDPProblemModel)
VLDataScienceMachineLearningPackage.backup(problem::MyMDPProblemModel, U::Array{Float64,1}, s::Int64)
VLDataScienceMachineLearningPackage.policy(Q_array::Array{Float64,2})
VLDataScienceMachineLearningPackage.Q(p::MyMDPProblemModel, U::Array{Float64,1})
VLDataScienceMachineLearningPackage.lookahead
VLDataScienceMachineLearningPackage.myrandpolicy(problem::MyMDPProblemModel, 
    world::MyRectangularGridWorldModel, s::Int)
VLDataScienceMachineLearningPackage.myrandstep(problem::MyMDPProblemModel, 
    world::MyRectangularGridWorldModel, s::Int, a::Int)
VLDataScienceMachineLearningPackage.iterative_policy_evaluation
VLDataScienceMachineLearningPackage.greedy
```
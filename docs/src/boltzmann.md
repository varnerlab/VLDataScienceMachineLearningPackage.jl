# Boltzmann Machines
A Boltzmann Machine is a type of stochastic recurrent neural network that can learn a probability distribution over its set of inputs. It consists of a network of symmetrically connected, neuron-like units that make stochastic decisions about whether to be on or off. Boltzmann Machines are particularly useful for unsupervised learning tasks, such as dimensionality reduction, feature learning, and generative modeling.

We have types and methods for working with classical Boltzmann Machines in the `VLDataScienceMachineLearningPackage.jl` package.

```@docs
VLDataScienceMachineLearningPackage.MySimpleBoltzmannMachineModel
VLDataScienceMachineLearningPackage.sample(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int})
```

## Restricted Boltzmann Machines (RBMs)
A Restricted Boltzmann Machine (RBM) is a special type of Boltzmann Machine that has a bipartite structure, meaning that its neurons are divided into two layers: a visible layer and a hidden layer. There are no connections between neurons within the same layer, which simplifies the learning algorithm and makes RBMs more efficient to train.

```@docs
VLDataScienceMachineLearningPackage.MyRestrictedBoltzmannMachineModel
VLDataScienceMachineLearningPackage.sample(model::MyRestrictedBoltzmannMachineModel, sₒ::Vector{Int})
VLDataScienceMachineLearningPackage.learn(model::MyRestrictedBoltzmannMachineModel, data::Array{Int64,2}, p::Categorical)
```
# Boltzmann Machines
A Boltzmann Machine is a type of stochastic recurrent neural network that can learn a probability distribution over its set of inputs. It consists of a network of symmetrically connected, neuron-like units that make stochastic decisions about whether to be on or off. Boltzmann Machines are particularly useful for unsupervised learning tasks, such as dimensionality reduction, feature learning, and generative modeling.

We have types and methods for working with Boltzmann Machines in the `VLDataScienceMachineLearningPackage.jl` package.

```@docs
VLDataScienceMachineLearningPackage.MySimpleBoltzmannMachineModel
VLDataScienceMachineLearningPackage.sample(model::MySimpleBoltzmannMachineModel, sₒ::Vector{Int})
```
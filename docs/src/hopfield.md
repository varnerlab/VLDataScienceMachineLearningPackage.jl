# Hopfield Networks
Hopfield Networks are a type of recurrent artificial neural network that serve as content-addressable memory systems with binary threshold nodes. They were invented by John Hopfield in 1982 and are used for pattern recognition and associative memory. This was one of the earliest models of what we think of today as a neural network and has influenced the development of more complex architectures, earning Hopfield a place in the history of artificial intelligence and a [Nobel Prize of Physics in 2024](https://www.nobelprize.org/prizes/physics/2024/hopfield/facts/).

We've encoded types and methods for both Classical and Modern Hopfield Networks in this package. Below, we provide an overview of how to use these models.

```@docs
VLDataScienceMachineLearningPackage.MyClassicalHopfieldNetworkModel
VLDataScienceMachineLearningPackage.MyModernHopfieldNetworkModel
VLDataScienceMachineLearningPackage.recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, trueenergyvalue::Float32)
recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
    maxiterations::Int64 = 1000, ϵₚ::Float64 = 1e-10, ϵₛ::Float64 = 1e-10) where T <: Number
```
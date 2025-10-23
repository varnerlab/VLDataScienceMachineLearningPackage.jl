# setup my paths -
const _PATH_TO_SRC = dirname(pathof(@__MODULE__));
const _PATH_TO_DATA = joinpath(_PATH_TO_SRC, "data");

# load external packages
using CSV
using DataFrames
using FileIO
using JSON
using DataStructures
using Distributions
using LinearAlgebra
using Statistics
using JuMP
using GLPK
using MadNLP

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Files.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Trees.jl"));
include(joinpath(_PATH_TO_SRC, "Wolfram.jl"));
include(joinpath(_PATH_TO_SRC, "Graphs.jl"));
include(joinpath(_PATH_TO_SRC, "Solvers.jl"));
include(joinpath(_PATH_TO_SRC, "Eigen.jl"));
include(joinpath(_PATH_TO_SRC, "Binary.jl"));
include(joinpath(_PATH_TO_SRC, "MDP.jl"));
include(joinpath(_PATH_TO_SRC, "Bandit.jl"));
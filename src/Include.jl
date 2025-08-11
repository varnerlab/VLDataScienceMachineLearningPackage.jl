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

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Files.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Trees.jl"));
include(joinpath(_PATH_TO_SRC, "Wolfram.jl"));
abstract type AbstractTextRecordModel end
abstract type AbstractTextDocumentCorpusModel end
abstract type AbstractFeatureHashingAlgorithm end
abstract type AbstractPriceTreeModel end
abstract type AbstractTreeModel end
abstract type AbstractRuleModel end
abstract type AbstractWolframSimulationAlgorithm end
abstract type AbstractGraphModel end
abstract type AbstractGraphNodeModel end
abstract type AbstractGraphEdgeModel end
abstract type AbstractGraphSearchAlgorithm end
abstract type AbstractGraphFlowAlgorithm end
abstract type AbstractGraphTraversalAlgorithm end
abstract type AbstractLinearSolverAlgorithm end
abstract type AbstractClassificationAlgorithm end
abstract type AbstractLinearProgrammingProblemType end
abstract type AbstractProcessModel end
abstract type AbstractWorldModel end

"""
    mutable struct MyLinearProgrammingProblemModel <: AbstractLinearProgrammingProblemType 

A mutable struct that represents a linear programming problem model.

### Fields
- `A::Array{Float64,2}`: constraint matrix
- `b::Array{Float64,1}`: right-hand side vector
- `c::Union{Array{Float64,2}, Array{Float64,1}}`: objective function coefficient matrix (vector)
- `lb::Array{Float64,1}`: lower bound vector
- `ub::Array{Float64,1}`: upper bound vector  

"""
mutable struct MyLinearProgrammingProblemModel <: AbstractLinearProgrammingProblemType
    
    # data -
    A::Array{Float64,2}     # constraint matrix
    b::Array{Float64,1}     # right-hand side vector
    c::Union{Array{Float64,2}, Array{Float64,1}}     # objective function coefficient matrix (vector)
    lb::Array{Float64,1}    # lower bound vector
    ub::Array{Float64,1}    # upper bound vector

    # constructor
    MyLinearProgrammingProblemModel() = new();
end



"""
    mutable struct MyPerceptronClassificationModel <: AbstractClassificationAlgorithm
A mutable struct that represents a perceptron classification model.

### Fields
    - `β::Vector{Float64}`: coefficients
    - `mistakes::Int64`: number of mistakes that are are willing to make
"""
mutable struct MyPerceptronClassificationModel <: AbstractClassificationAlgorithm
    
    # data -
    β::Vector{Float64}; # coefficients
    mistakes::Int64; # number of mistakes that are are willing to make

    # empty constructor -
    MyPerceptronClassificationModel() = new();
end

"""
    mutable struct MyLogisticRegressionClassificationModel <: AbstractClassificationAlgorithm

A mutable struct that represents a logistic regression classification model.

### Fields
    - `β::Vector{Float64}`: coefficients
    - `α::Float64`: learning rate
    - `ϵ::Float64`: convergence criterion
    - `L::Function`: loss function
"""
mutable struct MyLogisticRegressionClassificationModel <: AbstractClassificationAlgorithm
    
    # data -
    β::Vector{Float64}; # coefficients
    α::Float64; # learning rate
    ϵ::Float64; # convergence criterion
    L::Function; # loss function

    # empty constructor -
    MyLogisticRegressionClassificationModel() = new();
end


"""
    MySMSSpamHamRecordModel <: AbstractTextRecordModel

Model for a record in the SMS Spam Ham dataset.

### Fields
- `isspam::Bool` - a boolean value indicating if the message is spam.
- `message::String` - the content of the SMS message.
"""
mutable struct MySMSSpamHamRecordModel <: AbstractTextRecordModel
    
    # data -
    isspam::Bool
    message::String
    
    # constructor -
    MySMSSpamHamRecordModel() = new(); # empty
end

"""
    MySMSSpamHamRecordCorpusModel <: AbstractTextDocumentCorpusModel

Model for a collection of records in the SMS Spam Ham dataset.

### Fields
- `records::Dict{Int, MySMSSpamHamRecordModel}` - The records in the document (collection of records)
- `tokens::Dict{String, Int64}` - A dictionary of tokens in alphabetical order (key: token, value: position) for the entire document
"""
mutable struct MySMSSpamHamRecordCorpusModel <: AbstractTextDocumentCorpusModel
    
    # data -
    records::Dict{Int, MySMSSpamHamRecordModel}
    tokens::Dict{String, Int64}
    inverse::Dict{Int64, String}
    
    # constructor -
    MySMSSpamHamRecordCorpusModel() = new(); # empty
end

"""
    MySarcasmRecordModel <: AbstractTextRecordModel

Model for a record in the Sarcasm dataset.

### Fields
- `data::Array{String, Any}` - The data found in the record in the order they were found
"""
mutable struct MySarcasmRecordModel <: AbstractTextRecordModel
    
    # data -
    issarcastic::Bool
    headline::String
    article::String
    
    # constructor -
    MySarcasmRecordModel() = new(); # empty
end


"""
    MySarcasmRecordCorpusModel <: AbstractTextDocumentCorpusModel

Model for a collection of records in the Sarcasm dataset.

### Fields
- `records::Dict{Int, MySarcasmRecordModel}` - The records in the document (collection of records)
- `tokens::Dict{String, Int64}` - A dictionary of tokens in alphabetical order (key: token, value: position) for the entire document
"""
mutable struct MySarcasmRecordCorpusModel <: AbstractTextDocumentCorpusModel
    
    # data -
    records::Dict{Int, MySarcasmRecordModel}
    tokens::Dict{String, Int64}
    inverse::Dict{Int64, String}
    
    # constructor -
    MySarcasmRecordCorpusModel() = new(); # empty
end

struct UnsignedFeatureHashing <: AbstractFeatureHashingAlgorithm end
struct SignedFeatureHashing <: AbstractFeatureHashingAlgorithm end

"""
    mutable struct  MyAdjacencyRecombiningCommodityPriceTree <: AbstractPriceTreeModel

The `MyAdjacencyRecombiningCommodityPriceTree` type is a model of a commodity price tree that uses an dictionary to store the price data.
This model stores the connectivity information between nodes.

### Fields
- `data::Union{Nothing, Dict{Int64,NamedTuple}}` - A dictionary that stores the price data and path informationfor the tree.
- `connectivity::Dict{Int64,Array{Int64,1}}` - A dictionary that stores the connectivity information between nodes.
"""
mutable struct MyAdjacencyRecombiningCommodityPriceTree <: AbstractPriceTreeModel

    # data -
    data::Union{Nothing, Dict{Int64,NamedTuple}}
    connectivity::Dict{Int64,Array{Int64,1}}
    h::Int64 # height of the tree
    n::Int64 # branching factor

    # constructor 
    MyAdjacencyRecombiningCommodityPriceTree() = new()
end

""" 
    mutable struct MyFullGeneralAdjacencyTree <: AbstractTreeModel

The `MyFullGeneralAdjacencyTree` type is a model of a full general adjacency tree that uses a dictionary to store the tree structure.
There is a build and populate! method to build the tree and populate it with data.

### Fields
- `data::Union{Nothing, Dict{Int64,NamedTuple}}` - A dictionary that stores the node data for the tree.
- `connectivity::Dict{Int64,Array{Int64,1}}` - A dictionary that stores the connectivity information between nodes.
- `h::Int64` - The height of the tree.
- `n::Int64` - The branching factor of the tree.
"""
mutable struct MyFullGeneralAdjacencyTree <: AbstractTreeModel

    # data -
    data::Union{Nothing, Dict{Int64,NamedTuple}}
    connectivity::Dict{Int64,Array{Int64,1}}
    h::Int64 # height of the tree
    n::Int64 # branching factor

    # constructor
    MyFullGeneralAdjacencyTree() = new()
end

"""
    mutable struct MyOneDimensionalElementaryWolframRuleModel <: AbstractRuleModel

The `MyOneDimensionalElementarWolframRuleModel` mutable struct represents a one-dimensional elementary Wolfram rule model.

### Fields
- `index::Int` - The index of the rule
- `radius::Int` - The radius, i.e, the number of cells that influence the next state for this rule
- `rule::Dict{Int,Int}` - A dictionary that holds the rule where the `key` is the binary representation of the neighborhood and the `value` is the next state
"""
mutable struct MyOneDimensionalElementaryWolframRuleModel <: AbstractRuleModel
    
    # data
    index::Int
    radius::Int
    number_of_colors::Int
    rule::Dict{Int, Int}

    # constructor -
    MyOneDimensionalElementaryWolframRuleModel() = new();
end

struct WolframDeterministicSimulation <: AbstractWolframSimulationAlgorithm end
struct WolframStochasticSimulation <: AbstractWolframSimulationAlgorithm end

# -- GRAPHS BELOW HERE ---------------------------------------------------------------------------- #
"""
    mutable struct MyGraphNodeModel

A lightweight mutable node model used in simple graph representations.

### Fields
- `id::Int64` - Unique integer identifier for the node.
"""
mutable struct MyGraphNodeModel <: AbstractGraphNodeModel
   
   # data -
   id::Int64
   data::Union{Nothing, Any}; # this is a little fancy??

   # constructor -
   MyGraphNodeModel(id::Int64, data::Union{Nothing, Any}) = new(id, data);
end

"""
    mutable struct MyGraphEdgeModel

A mutable edge model representing a directed or undirected connection between nodes.
The model stores a numeric id, endpoint indices, and an optional numeric weight.

### Fields
- `id::Int64` - Unique integer identifier for the edge.
- `source::Int64` - Identifier of the source node (or one endpoint).
- `target::Int64` - Identifier of the target node (or the other endpoint).
- `weight::Union{Nothing, Any}` - Optional edge weight; nothing indicates an unweighted edge.
"""
mutable struct MyGraphEdgeModel <: AbstractGraphEdgeModel

    # data -
    id::Int64
    source::Int64
    target::Int64
    weight::Union{Nothing, Any}; # this is a little fancy??

    # constructor -
    MyGraphEdgeModel() = new();
end

"""
    mutable struct MyConstrainedGraphEdgeModel

Mutable model for a graph edge with capacity constraints. 
    
### Fields
- `id::Int64` - Unique identifier for the edge.
- `source::Int64` - Identifier for the source node.
- `target::Int64` - Identifier for the target node.
- `lower::Union{Nothing, Number}` - Lower capacity constraint for the edge.
- `upper::Union{Nothing, Number}` - Upper capacity constraint for the edge.

"""
mutable struct MyConstrainedGraphEdgeModel
    
    # data -
    id::Int64
    source::Int64
    target::Int64
    weight::Union{Nothing, Number}; # this is a little fancy??
    lower::Union{Nothing, Number}
    upper::Union{Nothing, Number}

    # constructor -
    MyConstrainedGraphEdgeModel() = new();
end


"""
    mutable struct MySimpleDirectedGraphModel

A minimal mutable directed graph container that keeps node and edge registries
and a children adjacency map for fast traversal of outgoing neighbors.

### Fields
- `nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}` - Optional mapping from node id to MyGraphNodeModel. Use nothing when uninitialized.
- `edges::Union{Nothing, Dict{Tuple{Int, Int}, Int64}}` - Optional mapping from (source, target) tuple to edge id. Use nothing when uninitialized.
- `children::Union{Nothing, Dict{Int64, Set{Int64}}}` - Optional adjacency map from a node id to the set of its child (outgoing) node ids.
"""
mutable struct MySimpleDirectedGraphModel <: AbstractGraphModel
   
   # data -
   nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}
   edges::Union{Nothing, Dict{Tuple{Int, Int}, Number}}
   children::Union{Nothing, Dict{Int64, Set{Int64}}}
   edgesinverse::Dict{Int, Tuple{Int, Int}} # map between edge id and source and target

   # constructor -
   MySimpleDirectedGraphModel() = new();
end

"""
    mutable struct MySimpleUndirectedGraphModel

A minimal mutable undirected graph container that keeps node and edge registries and a children adjacency map for fast traversal of outgoing neighbors.

### Fields
- `nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}` - Optional mapping from node id to MyGraphNodeModel. Use nothing when uninitialized.
- `edges::Union{Nothing, Dict{Tuple{Int, Int}, Int64}}` - Optional mapping from (source, target) tuple to edge id. Use nothing when uninitialized.
- `children::Union{Nothing, Dict{Int64, Set{Int64}}}` - Optional adjacency map from a node id to the set of its child (outgoing) node ids.
"""
mutable struct MySimpleUndirectedGraphModel <: AbstractGraphModel
   
    # data -
    nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}
    edges::Union{Nothing, Dict{Tuple{Int, Int}, Number}}
    children::Union{Nothing, Dict{Int64, Set{Int64}}}
    edgesinverse::Dict{Int, Tuple{Int, Int}} # map between edge id and source and target
 
    # constructor -
    MySimpleUndirectedGraphModel() = new();
end

"""
    mutable struct MyDirectedBipartiteGraphModel <: AbstractGraphModel

This type models a directed bipartite graph with source and sink nodes, along with maximum capacity constraints on the edges.
This type is constructed using a build method.

### Fields
- `nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}` - Optional mapping from node id to MyGraphNodeModel. Use nothing when uninitialized.
- `edges::Union{Nothing, Dict{Tuple{Int, Int}, Int64}}` - Optional mapping from (source, target) tuple to edge id. Use nothing when uninitialized.
- `children::Union{Nothing, Dict{Int64, Set{Int64}}}` - Optional adjacency map from a node id to the set of its child (outgoing) node ids.
- `edgesinverse::Dict{Int, Tuple{Int, Int}}` - Map between edge id and (source, target) tuple.
- `left::Set{Int64}` - Set of left (source) node ids.
- `right::Set{Int64}` - Set of right (sink) node ids.
- `source::Int64` - Source node id.
- `sink::Int64` - Sink node id.
- `capacity::Dict{Tuple{Int64, Int64}, Tuple{Number, Number}}` - Capacity constraints on the edges.
"""
mutable struct MyDirectedBipartiteGraphModel <: AbstractGraphModel
   
    # data -
    nodes::Union{Nothing, Dict{Int64, MyGraphNodeModel}}
    edges::Union{Nothing, Dict{Tuple{Int, Int}, Number}}
    children::Union{Nothing, Dict{Int64, Set{Int64}}}
    edgesinverse::Dict{Int, Tuple{Int, Int}} # map between edge id and source and target
    left::Set{Int64}
    right::Set{Int64}
    source::Int64
    sink::Int64
    capacity::Dict{Tuple{Int64, Int64}, Tuple{Number, Number}}

    # constructor -
    MyDirectedBipartiteGraphModel() = new();
end

# Let's create some tag types
struct DijkstraAlgorithm <: AbstractGraphSearchAlgorithm end
struct BellmanFordAlgorithm <: AbstractGraphSearchAlgorithm end
struct FordFulkersonAlgorithm <: AbstractGraphFlowAlgorithm end
struct EdmondsKarpAlgorithm <: AbstractGraphFlowAlgorithm end
struct DepthFirstSearchAlgorithm <: AbstractGraphTraversalAlgorithm end
struct BreadthFirstSearchAlgorithm <: AbstractGraphTraversalAlgorithm end
struct JacobiMethod <: AbstractLinearSolverAlgorithm end
struct GaussSeidelMethod <: AbstractLinearSolverAlgorithm end
struct SuccessiveOverRelaxationMethod <: AbstractLinearSolverAlgorithm end

# -- GRAPHS ABOVE HERE ---------------------------------------------------------------------------- #

# -- MDP AND RL BELOW HERE ------------------------------------------------------------------------ #


"""
    mutable struct MyMDPProblemModel <: AbstractProcessModel

A mutable struct that defines a Markov Decision Process (MDP) model. 
The MDP model is defined by the tuple `(𝒮, 𝒜, T, R, γ)`. 
The state space `𝒮` is an array of integers, the action space `𝒜` is an array of integers, 
the transition matrix `T` is a function or a 3D array, the reward matrix `R` is a function or a 2D array, 
and the discount factor `γ` is a float.

### Fields
- `𝒮::Array{Int64,1}`: state space
- `𝒜::Array{Int64,1}`: action space
- `T::Union{Function, Array{Float64,3}}`: transition matrix of function
- `R::Union{Function, Array{Float64,2}}`: reward matrix or function
- `γ::Float64`: discount factor
"""
mutable struct MyMDPProblemModel <: AbstractProcessModel

    # data -
    𝒮::Array{Int64,1}
    𝒜::Array{Int64,1}
    T::Union{Function, Array{Float64,3}}
    R::Union{Function, Array{Float64,2}}
    γ::Float64
    
    # constructor -
    MyMDPProblemModel() = new()
end

"""
    mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

A mutable struct that defines a rectangular grid world model.

### Fields
- `number_of_rows::Int`: number of rows in the grid
- `number_of_cols::Int`: number of columns in the grid
- `coordinates::Dict{Int,Tuple{Int,Int}}`: dictionary of state to coordinate mapping
- `states::Dict{Tuple{Int,Int},Int}`: dictionary of coordinate to state mapping
- `moves::Dict{Int,Tuple{Int,Int}}`: dictionary of state to move mapping
- `rewards::Dict{Int,Float64}`: dictionary of state to reward mapping
"""
mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

    # data -
    number_of_rows::Int
    number_of_cols::Int
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int},Int}
    moves::Dict{Int,Tuple{Int,Int}}
    rewards::Dict{Int,Float64}

    # constructor -
    MyRectangularGridWorldModel() = new();
end

"""
    struct MyValueIterationModel <: AbstractProcessModel

A struct that defines a value iteration model. 
The value iteration model is defined by the maximum number of iterations `k_max`.
"""
struct MyValueIterationModel 
    
    # data -
    k_max::Int64; # max number of iterations
end

"""
    struct MyValueFunctionPolicy

A struct that defines a value function policy.

### Fields
- `problem::MyMDPProblemModel`: MDP problem model
- `U::Array{Float64,1}`: value function vector. This holds the Utility of each state.
"""
struct MyValueFunctionPolicy
    problem::MyMDPProblemModel
    U::Array{Float64,1}
end


"""
    mutable struct MySimpleCobbDouglasChoiceProblem

A model for a Cobb-Douglas choice problem. 

### Fields
- `α::Array{Float64,1}`: the vector of parameters for the Cobb-Douglas utility function (preferences)
- `c::Array{Float64,1}`: the vector of unit prices for the goods
- `I::Float64`: the income the consumer has to spend
- `bounds::Array{Float64,2}`: the bounds on the goods [0,U] where U is the upper bound
- `initial::Array{Float64,1}`: the initial guess for the solution
"""
mutable struct MySimpleCobbDouglasChoiceProblem

    # data -
    α::Array{Float64,1}         # preferences
    c::Array{Float64,1}         # prices
    I::Float64                  # budget
    bounds::Array{Float64,2}    # bounds
    initial::Array{Float64,1}   # initial guess

    # constructor
    MySimpleCobbDouglasChoiceProblem() = new();
end
# -- MDP AND RL ABOVE HERE ------------------------------------------------------------------------ #
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
 
    # constructor -
    MySimpleUndirectedGraphModel() = new();
 end

struct DikjstraAlgorithm <: AbstractGraphSearchAlgorithm end
struct BellmanFordAlgorithm <: AbstractGraphSearchAlgorithm end
struct FordFulkersonAlgorithm <: AbstractGraphFlowAlgorithm end
struct DepthFirstSearchAlgorithm <: AbstractGraphTraversalAlgorithm end
struct BreadthFirstSearchAlgorithm <: AbstractGraphTraversalAlgorithm end

# -- GRAPHS ABOVE HERE ---------------------------------------------------------------------------- #
abstract type AbstractTextRecordModel end
abstract type AbstractTextDocumentCorpusModel end
abstract type AbstractFeatureHashingAlgorithm end
abstract type AbstractPriceTreeModel end
abstract type AbstractTreeModel end
abstract type AbstractRuleModel end
abstract type AbstractWolframSimulationAlgorithm end

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
- `records::Dict{Int, MySMSSpamHamRecordModel}`: The records in the document (collection of records)
- `tokens::Dict{String, Int64}`: A dictionary of tokens in alphabetical order (key: token, value: position) for the entire document
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
- `data::Array{String, Any}`: The data found in the record in the order they were found
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
- `records::Dict{Int, MySarcasmRecordModel}`: The records in the document (collection of records)
- `tokens::Dict{String, Int64}`: A dictionary of tokens in alphabetical order (key: token, value: position) for the entire document
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
- `data::Union{Nothing, Dict{Int64,NamedTuple}}`: A dictionary that stores the price data and path informationfor the tree.
- `connectivity::Dict{Int64,Array{Int64,1}}`: A dictionary that stores the connectivity information between nodes.
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
- `data::Union{Nothing, Dict{Int64,NamedTuple}}`: A dictionary that stores the node data for the tree.
- `connectivity::Dict{Int64,Array{Int64,1}}`: A dictionary that stores the connectivity information between nodes.
- `h::Int64`: The height of the tree.
- `n::Int64`: The branching factor of the tree.
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

### Required fields
- `index::Int`: The index of the rule
- `radius::Int`: The radius, i.e, the number of cells that influence the next state for this rule
- `rule::Dict{Int,Int}`: A dictionary that holds the rule where the `key` is the binary representation of the neighborhood and the `value` is the next state
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
abstract type AbstractTextRecordModel end
abstract type AbstractTextDocumentCorpusModel end
abstract type AbstractFeatureHashingAlgorithm end
abstract type AbstractPriceTreeModel end

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


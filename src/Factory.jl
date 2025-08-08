function _build(recordtype::Type{MySMSSpamHamRecordModel}, data::NamedTuple)::MySMSSpamHamRecordModel
    
    # get data from the NamedTuple -
    isspam = data.isspam;
    message = data.message;

    # clean the data - do NOT include puncuation in the headline -
    puncuation_skip_set = Set{Char}();
    push!(puncuation_skip_set, ',');
    push!(puncuation_skip_set, '.');
    push!(puncuation_skip_set, '!');
    push!(puncuation_skip_set, '?');
    push!(puncuation_skip_set, ';');
    push!(puncuation_skip_set, ':');
    push!(puncuation_skip_set, ')');
    push!(puncuation_skip_set, '(');
    push!(puncuation_skip_set, '\"');
    push!(puncuation_skip_set, '/');
    push!(puncuation_skip_set, '\\');
    push!(puncuation_skip_set, '-');
    push!(puncuation_skip_set, '_');
    push!(puncuation_skip_set, '`');
    push!(puncuation_skip_set, ''');
    push!(puncuation_skip_set, '*');
    push!(puncuation_skip_set, '+');
    push!(puncuation_skip_set, '=');
    push!(puncuation_skip_set, '@');
    push!(puncuation_skip_set, '%');
    push!(puncuation_skip_set, '|');
    push!(puncuation_skip_set, '{');
    push!(puncuation_skip_set, '}');
    push!(puncuation_skip_set, '[');
    push!(puncuation_skip_set, ']');
    push!(puncuation_skip_set, '<');
    push!(puncuation_skip_set, '>');
    push!(puncuation_skip_set, '~');
    push!(puncuation_skip_set, '^');
    push!(puncuation_skip_set, '&');
    push!(puncuation_skip_set, '$');
    push!(puncuation_skip_set, '¿');
    push!(puncuation_skip_set, '¡');
    push!(puncuation_skip_set, '£');
    push!(puncuation_skip_set, '€');
    push!(puncuation_skip_set, '¥');
    push!(puncuation_skip_set, '₹');   
    push!(puncuation_skip_set, '©'); 
    push!(puncuation_skip_set, '®');
    push!(puncuation_skip_set, '™');
    push!(puncuation_skip_set, '¯');
    push!(puncuation_skip_set, '\u00a0');

    # ok, so field is a string, and we are checking if it contains any of the puncuation characters
    chararray =  message |> collect;

    # let's use the filter function to remove any puncuation characters from the field -
    message = filter(c -> (c |> Int ) ≤ 255 && !(c ∈ puncuation_skip_set),
            chararray) |> String |> string-> strip(string, ' ') |> String;

    # checks?
    # if we split the message, we should have a list of words, with no field being empty
    fields = split(message, ' ') .|> String
    fields = filter(x -> x != "", fields)
    message = join(fields, ' ')

    # create the an empty instance of the modeltype, and then add data to it
    record = recordtype();
    record.isspam = isspam;
    record.message = message;
    
    # return the populated model -
    return record;
end


function _build(recordtype::Type{MySarcasmRecordModel}, data::NamedTuple)::MySarcasmRecordModel
    
    # get data from the NamedTuple -
    headlinerecord = data.headline;
    article = data.article;
    issarcastic = data.issarcastic;

    # clean the data - do NOT include puncuation in the headline -
    puncuation_skip_set = Set{Char}();
    push!(puncuation_skip_set, ',');
    push!(puncuation_skip_set, '.');
    push!(puncuation_skip_set, '!');
    push!(puncuation_skip_set, '?');
    push!(puncuation_skip_set, ';');
    push!(puncuation_skip_set, ':');
    push!(puncuation_skip_set, ')');
    push!(puncuation_skip_set, '(');
    push!(puncuation_skip_set, '\"');
    push!(puncuation_skip_set, '/');
    push!(puncuation_skip_set, '\\');
    push!(puncuation_skip_set, '-');
    push!(puncuation_skip_set, '_');
    push!(puncuation_skip_set, '`');
    push!(puncuation_skip_set, ''');
    push!(puncuation_skip_set, '*');
    push!(puncuation_skip_set, '+');
    push!(puncuation_skip_set, '=');
    push!(puncuation_skip_set, '@');
    push!(puncuation_skip_set, '%');
    push!(puncuation_skip_set, '|');
    push!(puncuation_skip_set, '{');
    push!(puncuation_skip_set, '}');
    push!(puncuation_skip_set, '[');
    push!(puncuation_skip_set, ']');
    push!(puncuation_skip_set, '<');
    push!(puncuation_skip_set, '>');
    push!(puncuation_skip_set, '~');
    push!(puncuation_skip_set, '^');
    push!(puncuation_skip_set, '&');
    push!(puncuation_skip_set, '$');
    push!(puncuation_skip_set, '¿');
    push!(puncuation_skip_set, '¡');
    push!(puncuation_skip_set, '£');
    push!(puncuation_skip_set, '€');
    push!(puncuation_skip_set, '¥');
    push!(puncuation_skip_set, '₹');   
    push!(puncuation_skip_set, '©'); 
    push!(puncuation_skip_set, '®');
    push!(puncuation_skip_set, '™');
    push!(puncuation_skip_set, '¯');
    push!(puncuation_skip_set, '\u00a0');

    # ok, so field is a string, and we are checking if it contains any of the puncuation characters
    chararray =  headlinerecord |> collect;

    # let's use the filter function to remove any puncuation characters from the field -
    headlinerecord = filter(c -> (c |> Int ) ≤ 255 && !(c ∈ puncuation_skip_set),
            chararray) |> String |> string-> strip(string, ' ') |> String;

    # checks?
    # if we split the headline, we should have a list of words, with no field being empty
    fields = split(headlinerecord, ' ') .|> String
    fields = filter(x -> x != "", fields)
    headlinerecord = join(fields, ' ')    

    # create the an empty instance of the modeltype, and then add data to it
    record = recordtype();
    record.headline = headlinerecord;
    record.article = article;
    record.issarcastic = issarcastic;
    
    # return the populated model -
    return record;
end


function build(record::Type{T}, data::NamedTuple)::T where T <: AbstractTextRecordModel 
    return _build(record, data);
end

"""
    function build(type::Type{MyAdjacencyRecombiningCommodityPriceTree}, data::NamedTuple) -> MyAdjacencyRecombiningCommodityPriceTree

Builds an `MyAdjacencyRecombiningCommodityPriceTree` model given the data in the `NamedTuple`. 
This method builds the connectivity of the tree. To compute the price at each node, use the `populate!` method.

### Arguments
- `type::Type{MyAdjacencyRecombiningCommodityPriceTree}`: The type of the model to build.
- `data::NamedTuple`: The data to use to build the model.

The `data` `NamedTuple` must contain the following fields:
- `h::Int64`: The height of the tree.
- `price::Float64`: The price at the root node.
- `u::Float64`: The price increase factor.
- `d::Float64`: The price decrease factor.

### Returns
- `MyAdjacencyRecombiningCommodityPriceTree`: the price tree model holding the computed price data.
"""
function build(modeltype::Type{MyAdjacencyRecombiningCommodityPriceTree}, 
    data::NamedTuple)::MyAdjacencyRecombiningCommodityPriceTree

    # get data -
    n = data.n; # branching factor
    h = data.h; # height of the tree
    model = modeltype(); # create an empty model
    
    # initialize -
    P = Dict{Int64,Float64}() 
    connectivity = Dict{Int64, Array{Int64,1}}()
    Nₕ = binomial(h + n, h) # number of nodes in the tree

    # main loop -
    for i ∈ 0:(Nₕ - 1)
        connectivity[i] = children_indices(i, n; base=0)
        (_, k) = index_counts(i, n)
        P[i] = 0.0; # default
    end

    
    # set the data, and connectivity for the model -
    model.data = P;
    model.connectivity = connectivity;
    model.h = h; # height of the tree
    model.n = n; # branching factor

    # return -
    return model;
end
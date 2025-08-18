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
    connectivity = Dict{Int64, Array{Int64,1}}()
    Nₕ = binomial(h + n, h) # number of nodes in the tree

    # main loop -
    for i ∈ 0:(Nₕ - 1)
        connectivity[i] = children_indices(i, n; base=0)
    end

    
    # set the data, and connectivity for the model -
    model.data = nothing; # we don't have any data yet, set as nothing
    model.connectivity = connectivity;
    model.h = h; # height of the tree
    model.n = n; # branching factor

    # return -
    return model;
end

"""
    function build(modeltype::Type{MyFullGeneralAdjacencyTree}, data::NamedTuple) -> MyFullGeneralAdjacencyTree

This function builds a `MyFullGeneralAdjacencyTree` model given the data in the `NamedTuple`. 
It populates the connectivity of the tree. However, it does not populate the data for the tree nodes.
We populate the data using the `populate!` method.

### Arguments
- `modeltype::Type{MyFullGeneralAdjacencyTree}`: The type of the model to build.
- `data::NamedTuple`: The data to use to build the model. The NamedTuple must have the following fields:
    - `h::Int64`: The height of the tree.
    - `n::Int64`: The branching factor of the tree.

### Returns
- `MyFullGeneralAdjacencyTree`: The constructed tree model.
"""
function build(modeltype::Type{MyFullGeneralAdjacencyTree}, 
    data::NamedTuple)::MyFullGeneralAdjacencyTree

    # get data -
    n = data.n; # branching factor
    h = data.h; # height of the tree
    model = modeltype(); # create an empty model
    
    # initialize -
    connectivity = Dict{Int64, Array{Int64,1}}()
    Nₕ = (n^(h+1) - 1) ÷ (n - 1); # number of nodes in the tree

    # main loop -
    for i ∈ 0:(Nₕ - 1)
        children = Array{Int64,1}();
        for k ∈ 1:n
            push!(children, n*i + k); # children indices
        end
        connectivity[i] = children;
    end

    
    # set the data, and connectivity for the model -
    model.data = nothing; # we don't have any data yet, set as nothing
    model.connectivity = connectivity;
    model.h = h; # height of the tree
    model.n = n; # branching factor

    # return -
    return model;
end

"""
    function build(modeltype::Type{MyOneDimensionalElementaryWolframRuleModel}, data::NamedTuple) -> MyOneDimensionalElementarWolframRuleModel

This `build` method constructs an instance of the [`MyOneDimensionalElementaryWolframRuleModel`](@ref) type using the data in a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple).

### Arguments
- `modeltype::Type{MyOneDimensionalElementaryWolframRuleModel}`: The type of model to build, in this case, the [`MyOneDimensionalElementaryWolframRuleModel`](@ref) type.
- `data::NamedTuple`: The data to use to build the model.

The `data::NamedTuple` must contain the following `keys`:
- `index::Int64`: The index of the Wolfram rule
- `colors::Int64`: The number of colors in the rule
- `radius::Int64`: The radius, i.e., the number of cells to consider in the rule

### Return
This function returns a populated instance of the [`MyOneDimensionalElementaryWolframRuleModel`](@ref) type.
"""
function build(modeltype::Type{MyOneDimensionalElementaryWolframRuleModel}, 
    data::NamedTuple)::MyOneDimensionalElementaryWolframRuleModel

    # initialize -
    index = data.index;
    colors = data.colors;
    radius = data.radius;

    # create an empty model instance -
    model = modeltype();
    rule = Dict{Int,Int}();

    # build the rule -
    number_of_states = colors^radius;
    states = digits(index, base=colors, pad=number_of_states);
    for i ∈ 0:number_of_states-1
        rule[i] = states[i+1];
    end
    
    # set the data on the object
    model.index = index;
    model.rule = rule;
    model.radius = radius;
    model.number_of_colors = colors;

    # return
    return model;
end

"""
    function build(model::Type{T}, edgemodels::Dict{Int64, MyGraphEdgeModel}) where T <: AbstractGraphModel

This function builds a graph model from a dictionary of edge models.

### Arguments
- `model::Type{T}`: The type of graph model to build, where `T` is a subtype of `AbstractGraphModel`.
- `edgemodels::Dict{Int64, MyGraphEdgeModel}`: A dictionary of edge models to use for building the graph.

### Returns
- `T`: The constructed graph model, where `T` is a subtype of `AbstractGraphModel`.
"""
function build(model::Type{T}, edgemodels::Dict{Int64, MyGraphEdgeModel}) where T <: AbstractGraphModel

    # build and empty graph model -
    graphmodel = model();
    nodes = Dict{Int64, MyGraphNodeModel}();
    edges = Dict{Tuple{Int64, Int64}, Number}();
    children = Dict{Int64, Set{Int64}}();
    edgesinverse = Dict{Int, Tuple{Int, Int}}();

    # let's build a list of nodes ids -
    tmp_node_ids = Set{Int64}();
    for (_,v) ∈ edgemodels
        push!(tmp_node_ids, v.source);
        push!(tmp_node_ids, v.target);
    end
    list_of_node_ids = tmp_node_ids |> collect |> sort;

    # remap the node ids to a contiguous ordering -
    nodeidmap = Dict{Int64, Int64}();
    nodecounter = 1;
    for id ∈ list_of_node_ids
        nodeidmap[id] = nodecounter;
        nodecounter += 1;
    end

    # build the nodes models -
    [nodes[nodeidmap[id]] = MyGraphNodeModel(nodeidmap[id], nothing) for id ∈ list_of_node_ids];

    # build the edges -
    for (_, v) ∈ edgemodels
        source_index = nodeidmap[v.source];
        target_index = nodeidmap[v.target];
        edges[(source_index, target_index)] = v.weight;
    end

    # build the inverse edge dictionary edgeid -> (source, target)
    n = length(nodes);
    edgecounter = 1;
    for source ∈ 1:n
        for target ∈ 1:n
            if haskey(edges, (source, target)) == true
                edgesinverse[edgecounter] = (source, target);
                edgecounter += 1;
            end
        end
    end
    
    # compute the children -
    for id ∈ list_of_node_ids
        newid = nodeidmap[id];
        node = nodes[newid];
        children[newid] = _children(edges, node.id);
    end


    # add stuff to model -
    graphmodel.nodes = nodes;
    graphmodel.edges = edges;
    graphmodel.edgesinverse = edgesinverse;
    graphmodel.children = children;

    # return -
    return graphmodel;
end

function build(edgemodel::Type{MyGraphEdgeModel}, data::NamedTuple)::MyGraphEdgeModel
   
    # initialize -
    model = edgemodel(); # build an empty edge model

    # get data from the tuple -
    id = data.id;
    source = data.source;
    target = data.target;
    weight = data.weight;
    
    # populate -
    model.id = id;
    model.source = source;
    model.target = target;
    model.weight = weight;

    # return -
    return model
end
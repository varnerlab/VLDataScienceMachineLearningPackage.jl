function _children(edges::Dict{Tuple{Int64, Int64}, Int64}, id::Int64)::Set{Int64}
    
    # initialize -
    childrenset = Set{Int64}();
    
    # Dumb implementation - why?
    for (k, _) ∈ edges
        if k[1] == id
            push!(childrenset, k[2]);
        end
    end

    # return -
    return childrenset;
end

# --- PUBLIC API BELOW HERE --------------------------------------------------------------------------- #

"""
    function children(graph::T, node::MyGraphNodeModel) -> Set{Int64} where T <: MyAbstractGraphModel

Returns the set of child node IDs for a given node in the graph.

### Arguments
- `graph::T`: The graph to search where `T <: MyAbstractGraphModel`.
- `node::MyGraphNodeModel`: The node to find children for.

### Returns
- `Set{Int64}`: The set of child node IDs.
"""
function children(graph::T, node::MyGraphNodeModel)::Set{Int64} where T <: MyAbstractGraphModel
    return graph.children[node.id];
end

"""
    function weight(graph::T, source::Int64, target::Int64, edgemodels::Dict{Int64, MyGraphEdgeModel}) -> Any where T <: MyAbstractGraphModel

Returns the weight of the edge between two nodes in the graph.

### Arguments
- `graph::T`: The graph to search where `T <: MyAbstractGraphModel`.
- `source::Int64`: The ID of the source node.
- `target::Int64`: The ID of the target node.

### Returns
- `Any`: The weight of the edge between the source and target nodes. We have this as `Any` to allow for flexibility in edge weights, which can be of any type.
"""
function weight(graph::T, source::Int64, target::Int64, edgemodels::Dict{Int64, MyGraphEdgeModel})::Any where T <: MyAbstractGraphModel
    
    # do a has key?
    if !haskey(graph.edges, (source, target))
        return nothing # or throw an error?
    end

    edge_id = graph.edges[(source, target)]
    if edge_id === nothing
        return nothing # or throw an error?
    end
    return edgemodels[edge_id].weight
end

# --- PUBLIC API ABOVE HERE --------------------------------------------------------------------------- #
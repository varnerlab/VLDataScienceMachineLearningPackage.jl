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


function _DFS(graph::T, node::MyGraphNodeModel, visited::Set{Int64}, order::Array{Int64,1}; 
    verbose::Bool = false) where T <: AbstractGraphModel

    # print - if verbose is true
    if (verbose == true)
        println("Visiting node: ", node.id);
    end

    if (in(node.id, visited) == false) # recursive case
        push!(visited, node.id); # add this node to the visited set -
        push!(order, node.id); # add this node to the order array

        mychildren = children(graph, node);  # get the children of the current node -

        # visit the children -
        for child in mychildren
            if (in(child, visited) == false) # mod from lecture pcode: don't recurse if the child has already been visited
                _DFS(graph, graph.nodes[child], visited, order, verbose=verbose); 
            end
        end
    end
end



function _BFS(graph::T, node::MyGraphNodeModel, visited::Set{Int64}, order::Array{Int64,1}; 
    verbose::Bool = false) where T <: AbstractGraphModel

    # initialize -
    q = Queue{Int64}();
    push!(order, node.id); # add this node to the order array

    # enqueue the first node -
    enqueue!(q, node.id);
    
    # main loop -
    while isempty(q) == false
        v = dequeue!(q);
        if (in(v,visited) == false)
            
            # print - if verbose is true
            if (verbose == true)
                println("Visiting node: $(v.id). My Queue: $(q)");
            end

            push!(visited, v); # add this node to the visited set
            push!(order, v); # add this node to the order array

            mychildren = children(graph, graph.nodes[v]);
            for child in mychildren
                if (in(child, visited) == false) # mod: don't enqueue if the child has already been visited
                    enqueue!(q, child);
                end
            end
        end
    end
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

"""
    function walk(graph::T, startnode::MyGraphNodeModel, algorithm::AbstractGraphTraversalAlgorithm; 
    verbose::Bool = false) where T <: AbstractGraphModel

The `walk` function traverses the graph starting from a given node using the specified algorithm (either Depth-First Search or Breadth-First Search). 
It maintains a set of visited nodes to avoid cycles and ensure that each node is processed only once.

### Arguments
- `graph::T`: The graph to traverse.
- `startnode::MyGraphNodeModel`: The node to start the traversal from.
- `algorithm::AbstractGraphTraversalAlgorithm`: The algorithm to use for the traversal (DFS or BFS).
- `verbose::Bool`: Whether to print verbose output (default is false).

### Returns
- `Array{Int64,1}`: The collection of visited node IDs in the order they were visited.
"""
function walk(graph::T, startnode::MyGraphNodeModel, algorithm::AbstractGraphTraversalAlgorithm; 
    verbose::Bool = false)::Array{Int64,1} where T <: AbstractGraphModel

    # initialize -
    visited = Set{Int64}();
    order = Array{Int64,1}();

    if algorithm isa DepthFirstSearchAlgorithm
        _DFS(graph, startnode, visited, order; verbose=verbose);
    elseif algorithm isa BreadthFirstSearchAlgorithm
        _BFS(graph, startnode, visited, order; verbose=verbose);
    else
        throw(ErrorException("Unsupported graph traversal algorithm"));
    end

    return order;
end

# --- PUBLIC API ABOVE HERE --------------------------------------------------------------------------- #
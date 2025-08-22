function _children(edges::Dict{Tuple{Int64, Int64}, Number}, id::Int64)::Set{Int64}
    
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

    # enqueue the first node -
    enqueue!(q, node.id);
    
    # main loop -
    while isempty(q) == false
        v = dequeue!(q);
        if (in(v,visited) == false)
            
            # print - if verbose is true
            if (verbose == true)
                println("Visiting node: $(v). My Queue: $(q)");
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

function _search(graph::T, start::MyGraphNodeModel, algorithm::DijkstraAlgorithm) where T <: AbstractGraphModel
    
    # initialize -
    distances = Dict{Int64, Float64}();
    previous = Dict{Int64, Union{Nothing,Int64}}();
    queue = PriorityQueue{Int64, Float64}(); # exported from DataStructures.jl

    # set distances and previous -
    distances[start.id] = 0.0; # distance from start to start is zero
    for (k, _) ∈ graph.nodes # what is this?
        if k != start.id
            distances[k] = Inf;
            previous[k] = nothing;
        end
        enqueue!(queue, k, distances[k]); # add nodes to the queue
    end

    # main loop -
    while !isempty(queue) # process nodes in the queue until it is empty (!isempty(queue) is the same as isempty(queue) == false)
        u = dequeue!(queue);
        mychildren = children(graph, graph.nodes[u]);

        for w ∈ mychildren # iterate over the children set of the current node
            alt = distances[u] + weight(graph, u, w); # distance to u so far + weight of edge from u to w
            if alt < distances[w] # Wow! the distance to w is less than the current best distance to w
                distances[w] = alt;
                previous[w] = u;
                queue[w] = alt;
            end
        end
    end

    # return -
    return distances, previous;
end

function _search(graph::T, start::MyGraphNodeModel, algorithm::BellmanFordAlgorithm) where T <: AbstractGraphModel

    # initialize -
    distances = Dict{Int64, Float64}();
    previous = Dict{Int64, Union{Nothing,Int64}}();
    nodes = graph.nodes;
    number_of_nodes = length(nodes);

    # initialize distance and previous dictionaries -
    for (_, node) ∈ nodes
        distances[node.id] = Inf;
        previous[node.id] = nothing;
    end
    distances[start.id] = 0.0;

    # main loop -
    counter = 1;
    while counter < (number_of_nodes - 1)
        
        for (k, _) ∈ graph.edges

            u = k[1];
            v = k[2];

            alt = distances[u] + weight(graph, u, v);
            if alt < distances[v]
                distances[v] = alt;
                previous[v] = u;
            end
        end

        # increment counter -
        counter += 1;
    end

    # check: If we have negatice cycles, then we should throw an error. 
    for (k, _) ∈ graph.edges

        u = k[1];
        v = k[2];

        if distances[u] + weight(graph, u, v) < distances[v]
            throw(ArgumentError("The graph contains a negative cycle"));
        end
    end

    # check fo
    return distances, previous;
end


# --- MAXIMUM FLOW API BELOW HERE --------------------------------------------------------------------- #
function _find_path_dfs(R::AbstractMatrix{T}, s::Int, t::Int; atol::T=zero(T)) where {T<:Real}
    n = size(R,1); size(R,2) == n || throw(ArgumentError("R must be square"))
    parent = fill(0, n)
    seen = falses(n)
    stack = Int[s]; seen[s] = true
    while !isempty(stack)
        u = pop!(stack)
        u == t && break
        @inbounds for v in 1:n
            if !seen[v] && R[u,v] > atol
                parent[v] = u
                seen[v] = true
                push!(stack, v)
            end
        end
    end
    if !seen[t]
        return false, parent, zero(T)
    end
    Δ = typemax(T)
    v = t
    while v != s
        u = parent[v]
        Δ = min(Δ, R[u,v])
        v = u
    end
    return true, parent, Δ
end

function _find_path_bfs(R::AbstractMatrix{T}, s::Int, t::Int; atol::T=zero(T)) where {T<:Real}
    n = size(R,1); size(R,2) == n || throw(ArgumentError("R must be square"))
    parent = fill(0, n)
    seen   = falses(n)
    q = Vector{Int}(undef, n); head = 1; tail = 1
    q[tail] = s; tail += 1
    seen[s] = true
    while head < tail
        u = q[head]; head += 1
        u == t && break
        @inbounds for v in 1:n
            if !seen[v] && R[u,v] > atol
                parent[v] = u
                seen[v] = true
                q[tail] = v; tail += 1
            end
        end
    end
    if !seen[t]
        return false, parent, zero(T)
    end
    Δ = typemax(T)
    v = t
    while v != s
        u = parent[v]
        Δ = min(Δ, R[u,v])
        v = u
    end
    return true, parent, Δ
end

function _augment_path!(R::AbstractMatrix{T}, F::AbstractMatrix{T},
                       parent::Vector{Int}, s::Int, t::Int, Δ::T) where {T<:Real}
    v = t
    while v != s
        u = parent[v]
        @inbounds begin
            R[u,v] -= Δ
            R[v,u] += Δ
            F[u,v] += Δ
            F[v,u] -= Δ
        end
        v = u
    end
    return nothing
end

function _edge_flows(C::AbstractMatrix, F::AbstractMatrix)
    n = size(C,1)
    flows = [(u, v, F[u,v]) for u in 1:n, v in 1:n if C[u,v] > 0 && F[u,v] > 0]
    sort!(flows)  # optional
    return flows
end

function _flow(algorithm::FordFulkersonAlgorithm, C::AbstractMatrix{T}, source::Int64, sink::Int64; 
    atol::Float64=1e-8)::Tuple{T, Dict{Tuple{Int64,Int64}, T}} where {T<:Real}

    # initialize -
    n = size(C,1)
    R = Matrix{T}(C)
    F = zeros(T, n, n)
    maxflow = zero(T)
    flows = Dict{Tuple{Int64,Int64}, T}();
    s = source;
    t = sink;

    # run the search -
    while true
        found, parent, Δ = _find_path_dfs(R, s, t; atol=atol)
        found || break
        _augment_path!(R, F, parent, s, t, Δ)
        maxflow += Δ
    end

    # populate the flows dictionary
    flow_tuple = _edge_flows(C, F);
    for (u, v, flow) ∈ flow_tuple
        flows[(u, v)] = flow
    end

    # returns a tuple of the maximum flow and the edge flows
    return (maxflow, flows)
end

function _flow(algorithm::EdmondsKarpAlgorithm, C::AbstractMatrix{T}, source::Int64, sink::Int64; 
    atol::Float64=1e-8)::Tuple{T, Dict{Tuple{Int64,Int64}, T}} where {T<:Real}

    # initialize -
    n = size(C,1)
    R = Matrix{T}(C)
    F = zeros(T, n, n)
    maxflow = zero(T)
    flows = Dict{Tuple{Int64,Int64}, T}();
    s = source;
    t = sink;

    # run the search -
    while true
        found, parent, Δ = _find_path_bfs(R, s, t; atol=atol)
        found || break
        _augment_path!(R, F, parent, s, t, Δ)
        maxflow += Δ
    end

    # populate the flows dictionary
    flow_tuple = _edge_flows(C, F);
    for (u, v, flow) ∈ flow_tuple
        flows[(u, v)] = flow
    end

    # returns a tuple of the maximum flow and the edge flows
    return (maxflow, flows)
end
# --- MAXIMUM FLOW API ABOVE HERE --------------------------------------------------------------------- #

# --- PUBLIC API BELOW HERE --------------------------------------------------------------------------- #

"""
    function children(graph::T, node::MyGraphNodeModel) -> Set{Int64} where T <: AbstractGraphModel

Returns the set of child node IDs for a given node in the graph.

### Arguments
- `graph::T`: The graph to search where `T <: AbstractGraphModel`.
- `node::MyGraphNodeModel`: The node to find children for.

### Returns
- `Set{Int64}`: The set of child node IDs.
"""
function children(graph::T, node::MyGraphNodeModel)::Set{Int64} where T <: AbstractGraphModel
    return graph.children[node.id];
end

"""
    function weight(graph::T, source::Int64, target::Int64, edgemodels::Dict{Int64, MyGraphEdgeModel}) -> Any where T <: AbstractGraphModel

Returns the weight of the edge between two nodes in the graph.

### Arguments
- `graph::T`: The graph to search where `T <: AbstractGraphModel`.
- `source::Int64`: The ID of the source node.
- `target::Int64`: The ID of the target node.

### Returns
- `Any`: The weight of the edge between the source and target nodes. We have this as `Any` to allow for flexibility in edge weights, which can be of any type.
"""
function weight(graph::T, source::Int64, target::Int64, edgemodels::Dict{Int64, MyGraphEdgeModel})::Any where T <: AbstractGraphModel
    
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
    function weight(graph::T, source::Int64, target::Int64) -> Float64 where T <: AbstractGraphModel

This function returns the weight of the edge between two nodes in a graph model.

### Arguments
- `graph::T`: the graph model to search. This is a subtype of `AbstractGraphModel`.
- `source::Int64`: the source node id.
- `target::Int64`: the target node id.

### Returns
- the weight of the edge between the source and target nodes.
"""
function weight(graph::T, source::Int64, target::Int64)::Float64 where T <: AbstractGraphModel   
    return graph.edges[(source, target)][1];
end

"""
    function walk(graph::T, startnode::MyGraphNodeModel, algorithm::AbstractGraphTraversalAlgorithm; 
    verbose::Bool = false) where T <: AbstractGraphModel

The `walk` function traverses the graph starting from a given node using the specified algorithm (either Depth-First Search or Breadth-First Search). 
It maintains a set of visited nodes to avoid cycles and ensure that each node is processed only once.

### Arguments
- `graph::T`: The graph to traverse.
- `startnode::MyGraphNodeModel`: The node to start the traversal from.
- `algorithm::AbstractGraphTraversalAlgorithm`: The algorithm to use for the traversal. This can be either an instance of `DepthFirstSearchAlgorithm` or `BreadthFirstSearchAlgorithm`. Default is `BreadthFirstSearchAlgorithm`.
- `verbose::Bool`: Whether to print verbose output (default is false).

### Returns
- `Array{Int64,1}`: The collection of visited node IDs in the order they were visited.
"""
function walk(graph::T, startnode::MyGraphNodeModel; 
    algorithm::AbstractGraphTraversalAlgorithm = BreadthFirstSearchAlgorithm(), 
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

"""
    findshortestpath(graph::T, start::MyGraphNodeModel; 
        algorithm::AbstractGraphSearchAlgorithm = BellmanFordAlgorithm()) where T <: AbstractGraphModel

The function computes the shortest paths from a starting node to all other nodes in a graph model. 

### Arguments
- `graph::T`: the graph model to search. This is a subtype of `AbstractGraphModel`.
- `start::MyGraphNodeModel`: the node to start the search from.
- `algorithm::MyAbstractGraphSearchAlgorithm`: the algorithm to use for the search. The default is `BellmanFordAlgorithm`, but it can also be `DijkstraAlgorithm`.

### Returns
- a tuple of two dictionaries: the first dictionary contains the distances from the starting node to all other nodes, and the second dictionary contains the previous node in the shortest path from the starting node to all other nodes.
"""
function findshortestpath(graph::T, start::MyGraphNodeModel;
    algorithm::AbstractGraphSearchAlgorithm = BellmanFordAlgorithm()) where T <: AbstractGraphModel
    return _search(graph, start, algorithm);
end

"""
    function maximumflow(graph::T, source::MyGraphNodeModel, sink::MyGraphNodeModel;
        algorithm::AbstractGraphFlowAlgorithm = FordFulkersonAlgorithm()) where T <: AbstractGraphModel

This function computes the maximum flow in a directed graph from a source node to a sink node using the specified algorithm.
The current implementation does not use the lower bound on the edge capacity (assumes all edges have a lower capacity bound of 0).

### Arguments
- `graph::T`: The graph to search. This needs to be a directed graph, with capacities on the edges. 
- `source::MyGraphNodeModel`: The source node.
- `sink::MyGraphNodeModel`: The sink node.
- `algorithm::AbstractGraphFlowAlgorithm`: The algorithm to use for the search. The default is `FordFulkersonAlgorithm`. The `EdmondsKarpAlgorithm` can also be used.
- `atol::Float64`: The absolute tolerance for floating point comparisons (default is 1e-8).

### Returns
- `Float64`: The maximum flow value.
- `Dict{{Int64,Int64}, Number}`: A dictionary mapping each edge (as a tuple of source and target node IDs) to its flow value.
"""
function maximumflow(graph::T, source::MyGraphNodeModel, sink::MyGraphNodeModel; 
    algorithm::AbstractGraphFlowAlgorithm = FordFulkersonAlgorithm(), atol::Float64 = 1e-8) where T <: AbstractGraphModel

    # both algorithms use the constraint matrix, so before we call the specific algorithm, we need to build that matrix from the graph model -
    n = length(graph.nodes); # how many nodes do we have?
    C = zeros(Float64, n, n); # create a capacity matrix
    for (edge, value) ∈ graph.capacity
        C[edge[1], edge[2]] = value[2]; # right now, we are ignoring the lower bound (value[1]) - TODO: fix this later
    end

    # get the ids for the source and sink nodes -
    s = source.id;
    t = sink.id;

    # call the flow calculation -
    return _flow(algorithm, C, s, t; atol=atol);
end

# --- PUBLIC API ABOVE HERE --------------------------------------------------------------------------- #
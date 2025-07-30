# -- PRIVATE METHODS BELOW HERE ------------------------------------------------------------------- #
function _featurehashing(algorithm::UnsignedFeatureHashing, text::Array{String,1}, d::Int64)::Array{Int64,1}

    # initialize -
    x = Dict{Int,Int}();
    size = d; # size of the hash table
    foreach(i-> x[i] = 0, range(0,step=1,length=size) |> collect)

    for s ∈ text
        h = hash(s);
        i = mod(h,size);
        x[i] += 1
    end

    # convert back to one-based array -
    result = zeros(size);
    for (k,v) ∈ x
        result[k+1] = v;
    end

    return result
end

function _featurehashing(algorithm::SignedFeatureHashing, text::Array{String,1}, d::Int64)::Array{Int64,1}

    # initialize -
    x = Dict{Int,Int}();
    size = d; # size of the hash table
    foreach(i-> x[i] = 0, range(0,step=1,length=size) |> collect)

    for s ∈ text
        h = hash(s);
        i = mod(h,size);
        if (h & 1 == 0)
            x[i] += 1;
        else
            x[i] -= 1;
        end
    end

    # convert back to one-based array -
    result = zeros(size);
    for (k,v) ∈ x
        result[k+1] = v;
    end

    return result
end

# -- PRIVATE METHODS ABOVE HERE ------------------------------------------------------------------- #

# -- PUBLIC METHODS BELOW HERE -------------------------------------------------------------------- #
"""
    function tokenize(s::String, tokens::Dict{String, Int64}; 
        pad::Int64 = 0, padleft::Bool = false, delim::Char = ' ') -> Array{Int64,1}

### Arguments
- `s::String` - the string to tokenize.
- `tokens::Dict{String, Int64}` - a dictionary of tokens in alphabetical order (key: token, value: position) for the entire document.
- `pad::Int64` - (optional) the number of padding tokens to add to the end of the tokenized string. Default is `0`.
- `padleft::Bool` - (optional) if `true`, the padding tokens are added to the beginning of the tokenized string. Default is `false`.
- `delim::Char` - (optional) the delimiter used in the string. Default is `' '`.

### Returns
- `Array{Int64,1}` - an array of integers representing the vectorized string.
"""
function tokenize(s::String, tokens::Dict{String, Int64}; 
    pad::Int64 = 0, padleft::Bool = false, delim::Char = ' ')::Array{Int64,1}

    # initialize -
    tokenarray = Array{Int64,1}();

    # control tokens -
    # push!(tokenarray, "<bos>");
    # push!(tokenarray, "<eos>");
    # push!(tokenarray, "<mask>");
    # push!(tokenarray, "<pad>");
    # push!(tokenarray, "<unk>"); # out of vocabulary

    # split the string -
    push!(tokenarray, tokens["<bos>"]); # beginning of sequence
    fields = split(s, delim) .|> String;
    for field ∈ fields
        if haskey(tokens, field)
            push!(tokenarray, tokens[field]);
        else
            push!(tokenarray, tokens["<unk>"]);
        end
    end

    # -- PAD LOGIC ----------------------------------------------------------- #
    # do we need to pad?
    if (padleft == false && pad > 0)
        N = length(tokenarray);
        foreach(i->push!(tokenarray, tokens["<pad>"]), (N+1):pad); # pad right
    elseif (padleft == true && pad > 0)
        N = length(tokenarray);
        foreach(i->pushfirst!(tokenarray, tokens["<pad>"]), (N+1):pad); # pad left
    end
    # ----------------------------------------------------------------------- #
    push!(tokenarray, tokens["<eos>"]); # end of sequence

    # return -
    return tokenarray;
end

"""
    function featurehashing(text::Array{String,1}; d::Int64 = 100, 
        algorithm::AbstractFeatureHashingAlgorithm = UnsignedFeatureHasing()) -> Array{Int64,1}

Computes the feature hashing of the input text using the specified algorithm.

### Arguments
- `text::Array{String,1}` - an array of strings to be hashed.
- `d::Int64` - (optional) the size of the hash table. Default is `100`.
- `algorithm::AbstractFeatureHasingAlgorithm` - (optional) the hashing algorithm to use. Default is `UnsignedFeatureHasing`.

### Returns
- `Array{Int64,1}` - an array of integers representing the hashed features.

"""
function featurehashing(text::Array{String,1}; d::Int64 = 100, 
    algorithm::AbstractFeatureHashingAlgorithm = UnsignedFeatureHashing())::Array{Int64,1}
    return _featurehashing(algorithm, text, d); # this will call the appropriate method based on the algorithm type
end

# -- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------- #
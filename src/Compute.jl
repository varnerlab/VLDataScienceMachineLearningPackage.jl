
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
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

function _featurehashing(algorithm::UnsignedFeatureHashing, text::Array{Int64,1}, d::Int64)::Array{Int64,1}

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

function _featurehashing(algorithm::SignedFeatureHashing, text::Array{Int64,1}, d::Int64)::Array{Int64,1}

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

"""
    function featurehashing(text::Array{Int,1}; d::Int64 = 100, 
        algorithm::AbstractFeatureHashingAlgorithm = UnsignedFeatureHasing()) -> Array{Int64,1}

Computes the feature hashing of the input text using the specified algorithm.

### Arguments
- `text::Array{Int,1}` - an array of integers to be hashed (e.g., tokenized text).
- `d::Int64` - (optional) the size of the hash table. Default is `100`.
- `algorithm::AbstractFeatureHasingAlgorithm` - (optional) the hashing algorithm to use. Default is `UnsignedFeatureHasing`.

### Returns
- `Array{Int64,1}` - an array of integers representing the hashed features.

"""
function featurehashing(text::Array{Int,1}; d::Int64 = 100, 
    algorithm::AbstractFeatureHashingAlgorithm = UnsignedFeatureHashing())::Array{Int64,1}
    return _featurehashing(algorithm, text, d); # convert to array of strings
end

"""
    function log_growth_matrix(dataset::Dict{String, DataFrame}, 
                firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0) -> Array{Float64,2}

The `log_growth_matrix` function computes the excess log growth matrix for a given set of firms where we define the log growth as:

```math
    \\mu_{t,t-1}(r_{f}) = \\frac{1}{\\Delta t} \\log\\left(\\frac{S_{t}}{S_{t-1}}\\right) - r_f
```

where ``S_t`` is the volume weighted average price (units: USD/share) at time `t`, ``\\Delta t`` is the time increment (in years), and ``r_f`` is the annual risk-free rate (units: 1/years) assuming
continuous compounding.

### Arguments
- `dataset::Dict{String, DataFrame}`: A dictionary of data frames where the keys are the firm ticker symbols and the values are the data frames holding price data. We use the `volume_weighted_average_price` column to compute the log growth by default.
- `firms::Array{String,1}`: An array of firm ticker symbols for which we want to compute the log growth matrix.
- `Δt::Float64`: The time increment used to compute the log growth. The default value is `1/252`, i.e., one trading day in units of years.
- `risk_free_rate::Float64`: The risk-free rate used to compute the log growth. The default value is `0.0`.
- `keycol::Symbol`: The column in the data frame to use to compute the log growth. The default value is `:volume_weighted_average_price`.
- `testfirm::String`: The firm ticker symbol to use to determine the number of trading days. By default, we use "AAPL".

### Returns
- `Array{Float64,2}`: An array of the excess log growth values for the given set of firms. The time series is the rows and the firms are the columns. The columns are ordered according to the order of the `firms` array.

### See:
* The `DataFrame` type (and methods for working with data frames) is exported from the [DataFrames.jl package](https://dataframes.juliadata.org/stable/)
"""
function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
    testfirm="AAPL", keycol::Symbol = :volume_weighted_average_price)::Array{Float64,2}

    # initialize -
    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset[testfirm]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    # main loop -
    for i ∈ eachindex(firms) 

        # get the firm data -
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        # compute the log returns -
        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, keycol];
            S₂ = firm_data[j, keycol];
            return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate;
        end
    end

    # return -
    return return_matrix;
end

"""
    vocabulary_transition_matrix(wordsdictionary::Dict{Char, Set{String}}, 
        letters::Array{Char,1}) -> Array{Float64,2}

This function computes the transition matrix for {a,b,c,...,z} given the vocabulary model and the letters.

### Arguments
- `wordsdictionary::Dict{Char, Set{String}}`: This is the vocabulary model that we have created.
- `letters::Array{Char,1}`: This is the array of characters that we are interested in.

### Returns
- `Array{Float64,2}`: This returns back the transition matrix.
"""
function vocabulary_transition_matrix(wordsdictionary::Dict{Char, Set{String}}, 
    letters::Array{Char,1})::Array{Float64,2}

    # initialize -
    number_of_letters = 26;
    counts = Array{Int64,1}(undef, number_of_letters);
    P = Array{Float64, 2}(undef, number_of_letters, number_of_letters);
    fill!(P, 0.0); # initialy all the elements are zero
    fill!(counts,0.0); # initially counts has all zeros

    # process each letter
    for i ∈ eachindex(letters)
        startchar = letters[i];
        
        # get all the words that start with the startchar -
        set_of_words = wordsdictionary[startchar] |> copy; # copy the set
        N = 0; # how many words have we processed?
        while (isempty(set_of_words) == false)
            word = pop!(set_of_words); # grab a word
            
            # get the second char of word -
            if (length(word) ≥ 2)
                next_char = word[2]; # get the second character in the word
                j = findfirst(x-> x == next_char, letters); # this gives me the index of the next_char
                counts[j] = counts[j] + 1;
                N += 1; # update the count of word samples that we explored
            end
        end
        
        # fill in the P -
        for k ∈ 1:number_of_letters
            P[i,k] = counts[k]/N;
        end

        # empty counts -
        fill!(counts,0.0); # fill it back up with zeros
    end

    # return the transition matrix -
    return P;
end

"""
    sample_words(P::Array{Float64,2}, letters::Array{Char,1};
        number_of_samples::Int64 = 100, length_of_sample_word::Int64 = 4, startchar::Char = 'z') -> Dict{Int64,String}

This function generates sample words of a specified length from the transition matrix P given the letters.

### Arguments
- `P::Array{Float64,2}`: This is the transition matrix.
- `letters::Array{Char,1}`: This is the array of characters that we are interested in.
- `number_of_samples::Int64`: This is the number of samples that we want to generate.
- `length_of_sample_word::Int64`: This is the length of the sample word that we want to generate.
- `startchar::Char`: This is the starting character of the sample word.

### Returns
- `Dict{Int64,String}`: This returns back a dictionary of sample words. The key is the index of the sample word and the value is the sample word.
"""
function sample_words(P::Array{Float64,2}, letters::Array{Char,1};
    number_of_samples::Int64 = 100, length_of_sample_word::Int64 = 4, 
    startchar::Char = 'z')::Dict{Int64,String}


    # initialize
    sample_word_start_char = startchar;
    letterindexmap = Dict{Int64,Char}()
    for i ∈ eachindex(letters)
        letterindexmap[i] = letters[i];
    end

    state_distribution_dictionary = Dict{Char,Categorical}();
    for i ∈ eachindex(letters)
        letter = letters[i];
        state_distribution_dictionary[letter] = Categorical(P[i,:]);
    end

    generated_word_dictionary = Dict{Int64,String}();
    for i ∈ 1:number_of_samples
        current_char = sample_word_start_char;
    
        wordvector = Array{Char,1}();
        push!(wordvector,current_char);
        for _ ∈ 2:length_of_sample_word
            d = state_distribution_dictionary[current_char];
            next_char = rand(d) |> k-> letterindexmap[k]; # returns back the char
            push!(wordvector,next_char);
            current_char = next_char;
        end

        generated_word_dictionary[i] = wordvector |> join
    end

    # return -
    return generated_word_dictionary;
end

# -- PUBLIC METHODS ABOVE HERE -------------------------------------------------------------------- #
# -- PRIVATE FUNCTIONS BELOW HERE ------------------------------------------------------------------------------ #
function _puzzleparse(filepath::String)::Vector{String}

    # initialize -
    result = Vector{String}();

    open(filepath, "r") do io
        for line ∈ eachline(io)
            push!(result, strip(line)); # cutoff whitespace
        end
    end
    return result;
end


function _jld2(path::String)::Dict{String,Any}
    return load(path);
end

_file_extension(file::String) = file[findlast(==('.'), file)+1:end]; # helper function to get the file extension

# -- PRIVATE FUNCTIONS ABOVE HERE ------------------------------------------------------------------------------ #

# -- PUBLIC FUNCTIONS BELOW HERE ------------------------------------------------------------------------------- #

"""
    MyTrainingMarketDataSet() -> Dict{String, DataFrame}

Load the components of the SP500 Daily open, high, low, close (OHLC) dataset as a dictionary of DataFrames.
This data was provided by [Polygon.io](https://polygon.io/) and covers the period from January 3, 2014, to December 31, 2024.

"""
MyTrainingMarketDataSet() = _jld2(joinpath(_PATH_TO_DATA, "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2"));


"""
    MyKaggleCustomerSpendingDataset() -> DataFrame

Load the Kaggle customer spending dataset as a DataFrame. 
The original dataset can be found at: [Spending dataset](https://www.kaggle.com/code/heeraldedhia/kmeans-clustering-for-customer-data?select=Mall_Customers.csv).
"""
function MyKaggleCustomerSpendingDataset()::DataFrame
    return CSV.read(joinpath(_PATH_TO_DATA, "mall-customers-dataset.csv"), DataFrame)
end

"""
    MyStringDecodeChallengeDataset() -> NamedTuple

Load the String Decode Challenge testing and production datasets. 

### Return
- `NamedTuple`: A tuple containing the three datasets:
    - `test_part_1`: The first part of the test dataset.
    - `test_part_2`: The second part of the test dataset.
    - `production`: The production dataset.
"""
function MyStringDecodeChallengeDataset()::NamedTuple

    # load three datasets -
    test_part_1 = _puzzleparse(joinpath(_PATH_TO_DATA, "test_part_1.txt"));
    test_part_2 = _puzzleparse(joinpath(_PATH_TO_DATA, "test_part_2.txt"));
    production_data = _puzzleparse(joinpath(_PATH_TO_DATA, "production.txt"));

    # package into a NamedTuple -
    data_tuple = (
        test_part_1 = test_part_1,
        test_part_2 = test_part_2,
        production = production_data
    );

    return data_tuple;
end

"""
    MyCommonSurnameDataset() -> DataFrame
Load the common surnames dataset by country as a DataFrame.
The original dataset can be found at: [Common Surnames by Country](https://github.com/sigpwned/popular-names-by-country-dataset).
"""
function MyCommonSurnameDataset()::DataFrame
    return CSV.read(joinpath(_PATH_TO_DATA, "common-surnames-by-country.csv"), DataFrame)
end

"""
    MyCommonForenameDataset() -> DataFrame
Load the common forenames dataset by country as a DataFrame.
The original dataset can be found at: [Common Forenames by Country](https://github.com/sigpwned/popular-names-by-country-dataset).
"""
function MyCommonForenameDataset()::DataFrame
    return CSV.read(joinpath(_PATH_TO_DATA, "common-forenames-by-country.csv"), DataFrame)
end


"""
    MyHousingPricesDataset() -> DataFrame

Load the house prices dataset from Kaggle as a DataFrame.
The original dataset can be found at: [Housing Prices Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?select=Housing.csv)
"""
function MyKaggleHousingPricesDataset()::DataFrame
    return CSV.read(joinpath(_PATH_TO_DATA, "Housing-Training-Dataset-Kaggle.csv"), DataFrame)
end

"""
    MyBanknoteAuthenticationDataset() -> DataFrame

The second dataset we will explore is the [banknote authentication dataset from the UCI archive](https://archive.ics.uci.edu/dataset/267/banknote+authentication). 
This dataset has `1372` instances of 4 continuous features and an integer (-1,1) class variable. 
"""
function MyBanknoteAuthenticationDataset()::DataFrame
    return CSV.read(joinpath(_PATH_TO_DATA, "data-banknote-authentication.csv"), DataFrame)
end

"""
    MyEnglishLanguageVocabularyModel() -> Dict{Char, Set{String}}

Load the English language vocabulary model as a dictionary where the keys are characters (the first letter of each word) 
and the values are sets of words that start with that letter.
"""
function MyEnglishLanguageVocabularyModel()::Dict{Char, Set{String}}

    # initialize -
    filepath = joinpath(_PATH_TO_DATA, "words_dictionary.json");
    data = JSON.parsefile(filepath); # read the data from the *.json file
    wordsdictionary = Dict{Char, Set{String}}(); # create an empty dictionary

    # the words are the keys of the dictionary
    list_of_words = keys(data) |> collect;
    for word ∈ list_of_words
        
        # what is the first letter of the word?
        first_letter = word[1]; # this gives the first letter of the word as a Char

        # do we have this letter in the model?
        if (haskey(wordsdictionary, first_letter) == false)
            wordsdictionary[first_letter] = Set{String}(); # create an empty new set
        end

        # add the word to the set
        push!(wordsdictionary[first_letter], word); # fancy!!
    end

    return wordsdictionary;
end

"""
    function MySarcasmCorpus() -> MySarcasmRecordCorpusModel

The function `corpus` reads a file composed of JSON records and returns the data as a `MySarcasmRecordCorpusModel` instance.
Each record in the file is expected to have the following fields:
* `is_sarcastic::Bool` - a boolean value indicating if the headline is sarcastic.
* `headline::String` - the headline of the article.
* `article_link::String` - the link to the article.


### Returns
- `MySarcasmRecordCorpusModel` - the data from the file as a `MySarcasmRecordCorpusModel` instance.

"""
function MySarcasmCorpus()::MySarcasmRecordCorpusModel

    # initialize the data -
    records = Dict{Int, MySarcasmRecordModel}();
    tokendictionary = Dict{String, Int64}();
    inverse = Dict{Int64, String}();
    counter = 1;

    # hard the path to the sarcasm dataset -
    path = joinpath(_PATH_TO_DATA, "Sarcasm_Headlines_Dataset_v2.txt");

    # open the file, process each line -
    open(path, "r") do io # open a stream to the file
        for line ∈ eachline(io)
            
            d = JSON.parse(line);
            records[counter] = build(MySarcasmRecordModel, (
                issarcastic = d["is_sarcastic"], headline = d["headline"], article = d["article_link"],
            ));

            counter += 1;
        end
    end

    # build the token dictionary -
    tokenarray = Array{String,1}();
    for (k,v) ∈ records

        # process headline data -
        headline = v.headline;
        tokens = split(headline, " ") .|> String;

        # process -
        for token ∈ tokens

            # strip any leading or trailing spaces -
            token = strip(token, ' ');
        
            if (in(token, tokenarray) == false && isempty(token) == false)
                push!(tokenarray, token);
            end
        end 
    end

    # build the token dictionary -
    tokenarray |> sort!
    for i ∈ eachindex(tokenarray)
        key = tokenarray[i]
        tokendictionary[key] = i - 1; 
        inverse[i - 1] = key;
    end

    # ok, so we need to update with the control tokens -
    controltokens = ["<bos>", "<eos>", "<mask>", "<pad>", "<unk>"];
    j = length(tokendictionary);
    for token ∈ controltokens
        if !(token in keys(tokendictionary))
            j += 1;
            tokendictionary[token] = j-1;
            inverse[j-1] = token;
        end
    end

    # set the data on the model -
    document = MySarcasmRecordCorpusModel();
    document.records = records;
    document.tokens = tokendictionary;
    document.inverse = inverse;

    # return -
    return document;
end

"""
    function MySMSSpamHamCorpus() -> MySMSSpamHamRecordCorpusModel

The function `MySMSSpamHamCorpus` reads the SMS Spam Ham dataset and returns the data as a `MySMSSpamHamRecordCorpusModel` instance.

"""
function MySMSSpamHamCorpus()::MySMSSpamHamRecordCorpusModel

    # initialize the data -
    records = Dict{Int, MySMSSpamHamRecordModel}();
    tokendictionary = Dict{String, Int64}();
    inverse = Dict{Int64, String}();
    counter = 1;

    # hard the path to the SMS Spam Ham dataset -
    path = joinpath(_PATH_TO_DATA, "sms-spam-ham-kaggle-cleaned.csv");

    # open the file, process each line -
    open(path, "r") do io # open a stream to the file
        for line ∈ eachline(io)
            d = split(line, ",") .|> String;

            # grab the label and message -
            label = d[1];
            message = d[2];

            # convert from spam/ham to boolean -
            isspam = (label == "spam") ? true : false;

            # build the record -
            records[counter] = _build(MySMSSpamHamRecordModel, (
                isspam = isspam, message = message,
            ));
            counter += 1;
        end
    end

    # build the token dictionary -
    tokenarray = Array{String,1}();
    for (k,v) ∈ records

        # process message data -
        message = v.message;
        tokens = split(message, " ") .|> String;

        # process -
        for token ∈ tokens

            # strip any leading or trailing spaces -
            token = strip(token, ' ');
        
            if (in(token, tokenarray) == false && isempty(token) == false)
                push!(tokenarray, token);
            end
        end 
    end

    # build the token dictionary -
    tokenarray |> sort!
    for i ∈ eachindex(tokenarray)
        key = tokenarray[i]
        tokendictionary[key] = i - 1; 
        inverse[i - 1] = key;
    end

    # ok, so we need to update with the control tokens -
    controltokens = ["<bos>", "<eos>", "<mask>", "<pad>", "<unk>"];
    j = length(tokendictionary);
    for token ∈ controltokens
        if !(token in keys(tokendictionary))
            j += 1;
            tokendictionary[token] = j-1;
            inverse[j-1] = token;
        end
    end

    # set the data on the model -
    document = MySMSSpamHamRecordCorpusModel();
    document.records = records;
    document.tokens = tokendictionary;
    document.inverse = inverse;

    # return -
    return document;
end

"""
    function MyGraphEdgeModels(filepath::String, edgeparser::Function; comment::Char='#', 
    delim::Char=',')::Dict{Int64,MyGraphEdgeModel}

Function to parse an edge file and return a dictionary of edges models.

### Arguments
- `filepath::String`: The path to the edge file.
- `edgeparser::Function`: A callback function to parse each edge line. This function should take a line as input, and a delimiter character, and return a tuple of the form `(source, target, data)`, where:
  - `source::Int64`: The source node ID.
  - `target::Int64`: The target node ID.
  - `data::Any`: Any additional data associated with the edge, e.g., a weight, a tuple of information, etc.

### Returns
- `Dict{Int64,MyGraphEdgeModel}`: A dictionary of edge models.
"""
function MyGraphEdgeModels(filepath::String, edgeparser::Function; comment::Char='#', 
    delim::Char=',')::Dict{Int64,MyGraphEdgeModel}

    # quick validation - ensure the path exists and is a regular file
    if !isfile(filepath)
        throw(ArgumentError("Invalid filepath: '$filepath' does not point to an existing regular file"))
    end

    # initialize
    edges = Dict{Int64, MyGraphEdgeModel}()
    linecounter = 0;
    
    # main -
    open(filepath, "r") do file # open a stream to the file
        for line ∈ eachline(file) # process each line in a file, one at a time
            
            # check: do we have comments?
            if (contains(line, comment) == true) || (isempty(line) == true)
                continue; # skip this line, and move to the next one
            end
            
            # # call the edge parser callback function -
            (s,t,data) = edgeparser(line, delim);

            # # build the edge model -
            edges[linecounter] = build(MyGraphEdgeModel, (
                source = s,
                target = t,
                weight = data,
                id = linecounter
            ));
            
            # update the line counter -
            linecounter += 1;
        end
    end

    # return -
    return edges;
end

"""
    function MyConstrainedGraphEdgeModels(filepath::String, edgeparser::Function; comment::Char='#', 
        delim::Char=',') -> Dict{Int64,MyConstrainedGraphEdgeModel}

This function parses a constrained graph edge file and returns a dictionary of constrained graph edge models.

### Arguments
- `filepath::String`: The path to the edge file.
- `edgeparser::Function`: A callback function to parse each edge line. This function should take a line as input, and a delimiter character, and return a tuple of the form `(source, target, weight, lower, upper)`, where:
  - `source::Int64`: The source node ID.
  - `target::Int64`: The target node ID.
  - `weight::Union{Nothing, Number}`: The weight of the edge.
  - `lower::Union{Nothing, Number}`: The lower bound of the edge weight.
  - `upper::Union{Nothing, Number}`: The upper bound of the edge weight.

### Returns
- `Dict{Int64,MyConstrainedGraphEdgeModel}`: A dictionary of constrained graph edge models.
"""
function MyConstrainedGraphEdgeModels(filepath::String, edgeparser::Function; comment::Char='#', 
    delim::Char=',')::Dict{Int64,MyConstrainedGraphEdgeModel}


    # quick validation - ensure the path exists and is a regular file
    if !isfile(filepath)
        throw(ArgumentError("Invalid filepath: '$filepath' does not point to an existing regular file"))
    end

    # initialize
    edges = Dict{Int64, MyConstrainedGraphEdgeModel}()
    linecounter = 0;
    
    # main -
    open(filepath, "r") do file # open a stream to the file
        for line ∈ eachline(file) # process each line in a file, one at a time
            
            # check: do we have comments?
            if (contains(line, comment) == true) || (isempty(line) == true)
                continue; # skip this line, and move to the next one
            end
            
            # # call the edge parser callback function -
            (s,t,w,l,u) = edgeparser(line, delim);

            # # build the edge model -
            edges[linecounter] = build(MyConstrainedGraphEdgeModel, (
                source = s,
                target = t,
                weight = w,
                lower = l,
                upper = u,
                id = linecounter
            ));
            
            # update the line counter -
            linecounter += 1;
        end
    end

    # return -
    return edges;
end

"""
    MyGrayscaleSimpsonsImageDataset() -> Dict{Int64, Array{Gray{N0f8},2}}

Load the Simpsons images dataset as a dictionary of grayscale images. This dataset contains 1000 images of Simpsons characters, each being 200 x 200 pixel images.
These images are taken from the [Simpsons Faces Dataset](https://www.kaggle.com/datasets/kostastokis/simpsons-faces) on Kaggle.

### Returns
- `Dict{Int64, Array{Gray{N0f8},2}}`: A dictionary where the keys are image indices and the values are 2D arrays representing grayscale images.
"""
function MyGrayscaleSimpsonsImageDataset()::Dict{Int64, Array{Gray{N0f8},2}}

    # initailize -
    training_image_dictionary = Dict{Int64, Array{Gray{N0f8},2}}();

    # hard the path to the Simpsons images dataset -
    pathtoimages = joinpath(_PATH_TO_DATA, "images-simpsons");

    # load the images -
    files = readdir(pathtoimages); 
    number_of_files = length(files);
    imagecount = 1;
    for i ∈ 1:number_of_files
        filename = files[i];
        ext = _file_extension(filename)
        if (ext == "png")
            training_image_dictionary[imagecount] = joinpath(pathtoimages, filename) |> x-> FileIO.load(x) |> img-> Gray.(img); # convert to grayscale
            imagecount += 1
        end
    end
    return training_image_dictionary;
end

"""
    MyUncorreleatedBlackAndWhiteImageDataset() -> Array{Gray{N0f8},3}

Load the uncorrelated black and white images dataset as a 3D array. This dataset contains 100 images of size 28 x 28 pixels.

### Returns
- `Array{Gray{N0f8},3}`: A 3D array where each slice along the third dimension represents a grayscale image.
"""
function MyUncorreleatedBlackAndWhiteImageDataset()::Array{Gray{N0f8},3}

    # initailize -
    number_of_rows = 28;
    number_of_cols = 28;
    number_of_training_examples = 100;
    image_digit_array = Array{Gray{N0f8},3}(undef, number_of_rows, number_of_cols, number_of_training_examples);

    # hard the path to the Simpsons images dataset -
    pathtoimages = joinpath(_PATH_TO_DATA, "images-uncorrelated-bw");
    files = readdir(pathtoimages);
    for i ∈ 1:(number_of_training_examples-1)    
        filename = files[i];
        image_digit_array[:,:,i] = joinpath(pathtoimages, filename) |> x-> FileIO.load(x);
    end
    return image_digit_array;
end

"""
    MyMNISTHandwrittenDigitImageDataset(; number_of_training_examples::Int64 = 1000) -> Dict{Int64, Array{Gray{N0f8},3}}

Load the MNIST digits dataset as a dictionary of grayscale images. This dataset contains images of handwritten digits (0-9), each being 28 x 28 pixel images.
The images were taken from the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

### Arguments
- `number_of_training_examples::Int64 = 1000`: The number of training examples to load for each digit (0-9). Default is 1000.

"""
function MyMNISTHandwrittenDigitImageDataset(; number_of_training_examples::Int64 = 1000)::Dict{Int64, Array{Gray{N0f8},3}}

    # initailize -
    number_of_rows = 28;
    number_of_cols = 28;
    number_digit_array = range(0, stop=9, step=1) |> collect;
    pathtoimages = joinpath(_PATH_TO_DATA, "images-mnist-digits");
    training_image_dictionary = Dict{Int64, Array{Gray{N0f8},3}}();

    # main loop -
    for i ∈ number_digit_array
        
        # create a set for this digit -
        image_digit_array = Array{Gray{N0f8},3}(undef, number_of_rows, number_of_cols, number_of_training_examples);
        files = readdir(joinpath(pathtoimages, "$(i)")); 
        imagecount = 1;
        for fileindex ∈ 1:number_of_training_examples
            filename = files[fileindex];
            ext = _file_extension(filename)
            if (ext == "jpg")
                image_digit_array[:,:,fileindex] = joinpath(pathtoimages, "$(i)", filename) |> x-> FileIO.load(x);
                imagecount += 1
            end
        end
    
        # capture -
        training_image_dictionary[i] = image_digit_array
    end

    return training_image_dictionary;
end
# -- PUBLIC FUNCTIONS ABOVE HERE ------------------------------------------------------------------------------ #
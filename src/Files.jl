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

    # add control tokens -
    push!(tokenarray, "<bos>");
    push!(tokenarray, "<eos>");
    push!(tokenarray, "<mask>");
    push!(tokenarray, "<pad>");
    push!(tokenarray, "<unk>"); # out of vocabulary

    tokenarray |> sort!
    for i ∈ eachindex(tokenarray)
        key = tokenarray[i]
        tokendictionary[key] = i - 1; 
        inverse[i - 1] = key;
    end

    # set the data on the model -
    document = MySarcasmRecordCorpusModel();
    document.records = records;
    document.tokens = tokendictionary;
    document.inverse = inverse;

    # return -
    return document;
end
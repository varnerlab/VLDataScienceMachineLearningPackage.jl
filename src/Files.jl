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
function MyKaggleCustomerSpendingDataset()::DataFrame
    return CSV.read(joinpath(_PATH_TO_DATA, "mall-customers-dataset.csv"), DataFrame)
end
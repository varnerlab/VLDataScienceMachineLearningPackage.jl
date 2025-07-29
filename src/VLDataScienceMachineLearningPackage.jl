module VLDataScienceMachineLearningPackage

# load the include file -
include("Include.jl");

# export data loading functions -
export MyKaggleCustomerSpendingDataset;
export MyStringDecodeChallengeDataset;
export MyCommonSurnameDataset;
export MyCommonForenameDataset;
export MySarcasmCorpus;

# types -
export AbstractTextRecordModel;
export AbstractTextDocumentCorpusModel;
export MySarcasmRecordModel;
export MySarcasmRecordCorpusModel;

# methods -
export tokenize;


end # module VLDataScienceMachineLearningPackage

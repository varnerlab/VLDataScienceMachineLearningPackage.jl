module VLDataScienceMachineLearningPackage

# load the include file -
include("Include.jl");

# export data loading functions -
export MyKaggleCustomerSpendingDataset;
export MyStringDecodeChallengeDataset;
export MyCommonSurnameDataset;
export MyCommonForenameDataset;
export MySarcasmCorpus;
export MySMSSpamHamCorpus;
export MyTrainingMarketDataSet;


# types -
export AbstractTextRecordModel;
export AbstractTextDocumentCorpusModel;
export AbstractPriceTreeModel;
export AbstractGraphTraversalAlgorithm;
export MySarcasmRecordModel;
export MySarcasmRecordCorpusModel;
export UnsignedFeatureHashing, SignedFeatureHashing;
export MySMSSpamHamRecordModel, MySMSSpamHamRecordCorpusModel;
export MyAdjacencyRecombiningCommodityPriceTree, MyFullGeneralAdjacencyTree;
export MyOneDimensionalElementaryWolframRuleModel;
export WolframDeterministicSimulation, WolframStochasticSimulation;
export MyGraphNodeModel, MyGraphEdgeModel, MyGraphEdgeModels, MySimpleDirectedGraphModel, MySimpleUndirectedGraphModel, DikjstraAlgorithm, BellmanFordAlgorithm, FordFulkersonAlgorithm;
export DepthFirstSearchAlgorithm, BreadthFirstSearchAlgorithm;


# methods -
export tokenize;
export featurehashing;
export build;
export populate!;
export simulate;
export children;
export weight;
export walk;


end # module VLDataScienceMachineLearningPackage

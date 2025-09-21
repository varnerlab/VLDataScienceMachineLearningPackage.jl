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
export MyKaggleHousingPricesDataset;
export MyBanknoteAuthenticationDataset;


# types -
# Abstract types -
export AbstractTextRecordModel;
export AbstractTextDocumentCorpusModel;
export AbstractFeatureHashingAlgorithm;
export AbstractPriceTreeModel;
export AbstractTreeModel;
export AbstractRuleModel;
export AbstractWolframSimulationAlgorithm;
export AbstractGraphModel;
export AbstractGraphNodeModel;
export AbstractGraphEdgeModel;
export AbstractGraphSearchAlgorithm;
export AbstractGraphFlowAlgorithm;
export AbstractGraphTraversalAlgorithm;
export AbstractLinearSolverAlgorithm;
export AbstractClassificationAlgorithm;
export AbstractLinearProgrammingProblemType;

# Concrete types -
export MyLinearProgrammingProblemModel;
export MyPerceptronClassificationModel;
export MySarcasmRecordModel;
export MySarcasmRecordCorpusModel;
export UnsignedFeatureHashing, SignedFeatureHashing;
export MySMSSpamHamRecordModel, MySMSSpamHamRecordCorpusModel;
export MyAdjacencyRecombiningCommodityPriceTree, MyFullGeneralAdjacencyTree;
export MyOneDimensionalElementaryWolframRuleModel;
export WolframDeterministicSimulation, WolframStochasticSimulation;
export MyGraphNodeModel, MyGraphEdgeModel, MyGraphEdgeModels, MySimpleDirectedGraphModel, MySimpleUndirectedGraphModel, MyDirectedBipartiteGraphModel, MyConstrainedGraphEdgeModel, MyConstrainedGraphEdgeModels;
export DepthFirstSearchAlgorithm, BreadthFirstSearchAlgorithm;
export DijkstraAlgorithm, BellmanFordAlgorithm, FordFulkersonAlgorithm, EdmondsKarpAlgorithm
export JacobiMethod, GaussSeidelMethod, SuccessiveOverRelaxationMethod;

# methods -
export tokenize;
export featurehashing;
export build;
export populate!;
export simulate;
export children;
export weight;
export walk;
export findshortestpath;
export maximumflow;
export solve;
export qriteration;
export log_growth_matrix;
export learn;
export classify;
export confusion;


end # module VLDataScienceMachineLearningPackage

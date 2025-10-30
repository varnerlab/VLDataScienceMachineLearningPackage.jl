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
export AbstractProcessModel;
export AbstractWorldModel;
export AbstractBanditAlgorithmModel;
export AbstractOnlineLearningModel;

# Concrete types -
export MyLinearProgrammingProblemModel;
export MyPerceptronClassificationModel, MyLogisticRegressionClassificationModel;
export MySarcasmRecordModel;
export MySarcasmRecordCorpusModel;
export MyEnglishLanguageVocabularyModel;
export UnsignedFeatureHashing, SignedFeatureHashing;
export MySMSSpamHamRecordModel, MySMSSpamHamRecordCorpusModel;
export MyAdjacencyRecombiningCommodityPriceTree, MyFullGeneralAdjacencyTree;
export MyOneDimensionalElementaryWolframRuleModel;
export WolframDeterministicSimulation, WolframStochasticSimulation;
export MyGraphNodeModel, MyGraphEdgeModel, MyGraphEdgeModels, MySimpleDirectedGraphModel, MySimpleUndirectedGraphModel, MyDirectedBipartiteGraphModel, MyConstrainedGraphEdgeModel, MyConstrainedGraphEdgeModels;
export DepthFirstSearchAlgorithm, BreadthFirstSearchAlgorithm;
export DijkstraAlgorithm, BellmanFordAlgorithm, FordFulkersonAlgorithm, EdmondsKarpAlgorithm
export JacobiMethod, GaussSeidelMethod, SuccessiveOverRelaxationMethod;
export MyValueIterationModel, MyValueFunctionPolicy, MyRandomRolloutModel;
export MyRectangularGridWorldModel, MyMDPProblemModel, MySimpleCobbDouglasChoiceProblem;
export MyExploreFirstAlgorithmModel, MyEpsilonGreedyAlgorithmModel, MyUCB1AlgorithmModel;
export MyBinaryWeightedMajorityAlgorithmModel, MyTwoPersonZeroSumGameModel;

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
export vocabulary_transition_matrix;
export sample_words;

# MDP and RL methods -
export lookahead;
export backup;
export Q;
export policy;
export myrandpolicy;
export myrandstep;
export iterative_policy_evaluation;
export greedy;

# bandit methods -
export regret;

# WMA and MWA methods -
export play;

end # module VLDataScienceMachineLearningPackage

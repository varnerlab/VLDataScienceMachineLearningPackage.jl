# Working with Text Data
We'll work with text data in many applications. We've included a few methods to help with text processing. 

```@docs
VLDataScienceMachineLearningPackage.tokenize
VLDataScienceMachineLearningPackage.featurehashing
```

## English Language Vocabulary Model
We have included a simple vocabulary model that we can use to analyze text data. This model is based on a 
dictionary of common English words. We can use this model to compute a transition matrix that describes the 
probability of transitioning from one letter to another in a word. This can be useful for various applications, 
including text generation and analysis.

```@docs
VLDataScienceMachineLearningPackage.vocabulary_transition_matrix
VLDataScienceMachineLearningPackage.sample_words
```
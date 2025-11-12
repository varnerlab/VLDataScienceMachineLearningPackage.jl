# Q-Learning
Q-Learning is a model-free reinforcement learning algorithm that aims to learn the value of taking a particular action in a given state. It does this by iteratively updating a Q-value table based on the agent's experiences in the environment.

```@docs
VLDataScienceMachineLearningPackage.MyQLearningAgentModel
VLDataScienceMachineLearningPackage.solve(model::MyQLearningAgentModel, environment::MyRectangularGridWorldModel; maxsteps::Int = 100, δ::Float64 = 0.02, worldmodel::Function = _world)
```
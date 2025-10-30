# Multiplicative Weight Updates
This section describes the implementation of the Binary Weighted Majority Algorithm (WMA) and Multiplicative Weights Algorithm (MWA) for online learning in the `VLDataScienceMachineLearningPackage.jl` Julia package.

```@docs
VLDataScienceMachineLearningPackage.MyBinaryWeightedMajorityAlgorithmModel
VLDataScienceMachineLearningPackage.play
```

## Zero-Sum Game Setting
Let's consider the application of the multiplicative weights update algorithm to zero-sum games. 

> In [a zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game), players have _opposing interests_, and the players' payoffs sum to zero: one's gain is the other's loss. The multiplicative-weights (MW) algorithm finds (approximate) Nash equilibria by down-weighting poorly performing actions over repeated play.

Let's dig into some the details of the game:
* **Game**: Consider a competitive setting with $k$ players. A game is called **zero-sum** if, for any outcome, the players' payoffs add to zero. The standard theory we use below focuses on the $k = 2$ case. Each player chooses an action $a \in \mathcal{A}$ from some finite action set $\mathcal{A}$ with $|\mathcal{A}| = N$. For the two-player case, we model payoffs with a matrix $\mathbf{M} \in \mathbb{R}^{N \times N}$ (for simplicity, assume both players have $N$ actions). If the row player chooses action $i$ and the column player chooses action $j$, then the row player's payoff is $m_{ij}$ and the column player's payoff is $-m_{ij}$. This is what we mean by __zero-sum__: whatever one player gains, the other loses.

* **Goals**: The row player wants to **maximize** their payoff. The column player wants to **minimize** the row player's payoff. Let the row player randomize over rows using a mixed strategy $\mathbf{p}$ (a probability distribution over the $N$ rows), and let the column player randomize over columns using a mixed strategy $\mathbf{q}$ (a probability distribution over the $N$ columns). The expected payoff to the row player is $\mathbf{p}^{\top}\mathbf{M}\mathbf{q}$ and because the game is zero-sum, the expected payoff to the column player is $-\mathbf{p}^{\top}\mathbf{M}\mathbf{q}$. So both players care about the same scalar $\mathbf{p}^{\top}\mathbf{M}\mathbf{q}$, but they pull it in opposite directions.

* **Nash Equilibrium**: A Nash equilibrium is a pair of (possibly mixed) strategies $(\mathbf{p}^*, \mathbf{q}^*)$ such that each player's strategy is a best response to the other's. In other words, given $\mathbf{q}^*$, the row player cannot switch from $\mathbf{p}^*$ to some other $\mathbf{p}$ and improve their expected payoff, and given $\mathbf{p}^*$, the column player cannot switch from $\mathbf{q}^*$ to some other $\mathbf{q}$ and further reduce the row player's expected payoff.

In a two-player zero-sum game, every Nash equilibrium corresponds to a **minimax solution**. The minimax theorem guarantees that:
$$
\max_{\mathbf{p}} \min_{\mathbf{q}} \mathbf{p}^{\top}\mathbf{M}\mathbf{q} = \min_{\mathbf{q}} \max_{\mathbf{p}} \mathbf{p}^{\top}\mathbf{M}\mathbf{q} = v
$$
where $v$ is called the value of the game. At equilibrium, the row player's strategy $\mathbf{p}^*$ guarantees at least $v$ no matter what the column player does, and the column player's strategy $\mathbf{q}^*$ holds the row player to at most $v$ no matter what the row player does. That shared value $v$ is the Nash equilibrium payoff.
  
Finally, learning dynamics: if both players repeatedly play the game and update their mixed strategies using sublinear algorithms such as multiplicative weights, then the time-averaged strategies approach an $\epsilon$-Nash equilibrium (equivalently, an $\epsilon$-minimax solution), where $\epsilon$ becomes small as regret becomes small.


### Algorithm
Let's outline a simple implementation of the multiplicative weights update algorithm for a two-player zero-sum game. Given a payoff matrix $\mathbf{M}\in\mathbb{R}^{N\times{N}}$, we want to find a _mixed strategy_, a probability distribution over actions, for the row player that minimizes expected loss.

__Initialization:__ Given a payoff matrix $\mathbf{M}\in\mathbb{R}^{N\times{N}}$, where the payoffs (elements of $\mathbf{M}$) are in the range $m_{ij}\in[-1, 1]$. 
Initialize the weights $w_{i}^{(1)} \gets 1$ for all actions $i\in\mathcal{A}$, where $\mathcal{A} = \{1,2,\dots,N\}$, and set the learning rate $\eta\in(0,1)$.

> __Choosing T__: The number of rounds $T$ determines the accuracy of the approximate Nash equilibrium. To achieve an $\epsilon$-Nash equilibrium, choose $T \geq \frac{\ln N}{\epsilon^2}$. For example, with $N=10$ actions and desired accuracy $\epsilon=0.1$, we need $T \geq \frac{\ln 10}{0.01} \approx 230$ rounds.

> __Choosing η__: The learning rate $\eta$ controls the step size of weight updates. Common rules of thumb include:
> - __Theory-based__: $\eta = \sqrt{\frac{\ln N}{T}}$ optimizes the convergence bound
> - __Simple rule__: $\eta = \frac{1}{\sqrt{T}}$ for practical applications  
> - __Adaptive__: Start with $\eta = 0.1$ and reduce by half if convergence stalls
> - __Constraint__: Ensure $\eta \leq 1$ to prevent negative weights (since losses are bounded in $[-1,1]$)

For each round $t=1,2,\dots,T$ __do__:
1. Compute the normalization factor: $\Phi^{(t)} \gets \sum_{i=1}^{N}w_{i}^{(t)}$.
1. __Row player__ computes its strategy: The _row player_ will choose an action with probability $\mathbf{p}^{(t)} \gets \left\{w_{i}^{(t)}/\Phi^{(t)} \mid i = 1,2,\dots,N\right\}$. Let the row player action be $i^{\star}$.
2. __Column player__ computes its strategy: The _column player_ will choose action: $j\gets \text{arg}\min_{j\in\mathcal{A}}\left\{\mathbf{p}^{(t)\top}\mathbf{M}\mathbf{e}_{j}\right\}$, so that $\mathbf{q}^{(t)} \gets \mathbf{e}_{j}$, where $\mathbf{e}_{j}$ is the $j$-th standard basis vector. The row player experiences loss vector $\boldsymbol{\ell}^{(t)} \gets \mathbf{L}\mathbf{q}^{(t)}$, where $\mathbf{L} = -\mathbf{M}$ is the loss matrix.
3. Update the weights: $w_i^{(t+1)} \gets w_i^{(t)}\;\exp\bigl(-\eta\,\ell_i^{(t)}\bigr)$ for all actions $i\in\mathcal{A}$ for the row player.

### Convergence
After $T$ rounds, define the average strategies:  
$$
\bar p \;=\;\frac{1}{T}\sum_{t=1}^{T}p^{(t)}, 
\quad
\bar q \;=\;\frac{1}{T}\sum_{t=1}^{T}q^{(t)}.
$$
Then $(\bar p,\bar q)$ is an $\epsilon$-Nash equilibrium with
$$
  \max_{q}\,\bar p^\top M\,q
  \;-\;\min_{p}\,p^\top M\,\bar q
  \;\le\;\epsilon,
  \quad
  \epsilon = O\Bigl(\sqrt{\tfrac{\ln N}{T}}\Bigr).
$$

```@docs
VLDataScienceMachineLearningPackage.MyTwoPersonZeroSumGameModel
VLDataScienceMachineLearningPackage.play(model::MyTwoPersonZeroSumGameModel)
```
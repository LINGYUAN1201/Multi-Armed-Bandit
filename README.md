## Multi-Armed Bandit Algorithms
This repository contains a Python implementation of several multi-armed bandit (MAB) algorithms, designed for research and educational purposes.

## Features
BernoulliBandit Class: Simulates a multi-armed bandit problem with Bernoulli-distributed rewards.
EpsilonGreedy Class: Implements the Epsilon-Greedy algorithm, balancing exploration and exploitation.
UCB Class: Implements the Upper Confidence Bound (UCB) algorithm, using uncertainty to guide exploration.
ThompsonSampling Class: Implements the Thompson Sampling algorithm, leveraging Bayesian inference.
plot_results Function: Visualizes the cumulative regret over time for different MAB algorithms.

## Dependencies
numpy
matplotlib
seaborn
Install the dependencies using pip.

## Usage
Initialize the Bernoulli multi-armed bandit.
Run the algorithms (Epsilon-Greedy, UCB, Thompson Sampling).
Plot and compare the results to analyze the performance.
Example Output

The output includes cumulative regret metrics for each algorithm, demonstrating their effectiveness over multiple iterations.




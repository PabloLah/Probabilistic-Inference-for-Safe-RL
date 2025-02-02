### Probabilistic Inference for Safe Reinforcement Learning

This repository is a fork of an existing project where I contributed by modifying the algorithm to implement a new version of a Safe Reinforcement Learning method, following an idea from Jonas Hübotter and using a codebase provided by Yarden As. Specifically, I extended the Soft Actor-Critic (SAC) algorithm by incorporating a probabilistic inference-based approach to ensure safety constraints are met. This adaptation, referred to as **SafeSAC**, introduces a new formulation of the reinforcement learning problem using entropy-regularized objectives and safety-aware decision-making.

The work builds upon an existing SAC implementation and involved adjustments to the learning process, safety critics, and classifier-based state evaluations. The project was part of my semester research at ETH Zurich, where I explored probabilistic methods in reinforcement learning under safety constraints. My modifications focused on integrating safety guarantees without compromising learning efficiency, implementing novel optimization objectives, and addressing challenges related to classifier training and uncertainty estimation.

For a deeper understanding of the theoretical background and implementation details, please refer to my **semester project report**, which outlines the motivation, challenges, and future directions for this approach.

## Getting Started

### Setting up your environment

1. Make sure to have a clean virtual environment with python 3.10.
2. `pip install -r requirements.txt`

### Running experiments

- Running some defaults:
  `python experiment.py`

- Running the pendulum experiment:
  `python experiment.py +experiment=pendulum`

- Debugging:
  `python experiment.py +experiment=debug`

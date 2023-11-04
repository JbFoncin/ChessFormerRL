# ChessFormerRL

A repos to explore reinforcement learning using transformers with chess game.

## Finished

The vanilla DQN is finished with sequential training, with GPU support. trained models will be uploaded soon

The DQN v2 is finished! The training is faster than the previous version, but some improvement didn't work.

For example, the multi step reward is a failure. The models still have a loss increasing after a long training time.

By the way the agent is still able to increase efficiently the rewards.

The DQN v3, using quantile regression, does not converge. After a while, the agent sticks to bad rewards actions

## Roadmap

The next step is to implement the reinforce algorithm to compare policy based and value based strategies.

# ChessFormerRL

A repos to explore reinforcement learning using transformers with chess game.

## Finished

The vanilla DQN is finished with sequential training, with GPU support. trained models will be uploaded soon

The DQN v2 is finished! I am still testing it but results look good.

I had to use a lot of tricks to make this version running in usable times. I had to do some choices making the code a bit strange right now but explanations will be added in comments. Basically, I had to compute the sampling probabilities on GPU, which may look weird but works really well.

## Roadmap

DQN v3 : add QR-DQN

Actor-Critic : A2C A3C

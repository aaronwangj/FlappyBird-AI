# FlappyBird-AI

An AI that plays Flappy Bird. The AI is powered by a feedforward neural network created using an evolutionary algorithm, NEAT (NeuroEvolution of Augmenting Topologies). In collaboration with Tim Ruscica.

## NEAT Overview
In short, the algorithm mimics Darwin's natural selection by assigning fitness scores to each bird; these scores are incremented based on how long each bird lives, and those with the highest fitness scores play in the next generation with a set mutation rate. As generations progress, the best artificial neural network architectures and weights that lead to the highest fitness score will be discovered and used.

## Results
The AI will usually find an optimal neural network within five generations, but may take up to ten and as little as three generations.

## Requirements
Pygame, NEAT, and Python 3.7.

## Instructions
Download the files, run flappybird.py, and watch the AI train itself to conquer Flappy Bird. Enjoy!

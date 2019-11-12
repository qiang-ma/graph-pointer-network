# Combinatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning

## Dependencies
Python>=3.6

PyTorch=1.1


## Baselines
Code for running baselines.


## TSP_small
Code, data, and model for small scale travelling salesman problem (TSP). To train the model, please run train.py via
```
python train.py --size=X --epoch=X --batch_size=X --train_size=X --val_size=X --lr=X
```
Here the parameter --size is the size of TSP instance, and --lr is the learning rate. To test the model with data generated on the fly, please run test_random.py via
```
python test_random.py --size=X --batch_size=X --test_size=X --test_steps=X
```
To test the model with heldout TSP data, please run test.py via
```
python test.py --size=X
```

## TSP_larger
In this experiment, we train the model with small instances and use the model to predict the routes for larger scale TSP, i.e. TSP250/500. Please run the ipython notebook.


## TSPTW
In this experiment, we use hierarchical reinforcement learning to tackle TSP with Time Window. To train hierarchical model, please first train the lower model by
```
python tsptw_low.py
```
Then train higher model by 
```
python tsptw_high.py
```
To train non-hierarchical model, use
```
python tsptw_non_hier.py
```
To test hierarchical model using greedy method, use
```
python test_hier.py
```
To test hierarchical model using sampling method, use
```
python test_hier_sampling.py
```
To test non-hierarchical model, use
```
python test.py
```

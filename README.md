# Combinatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning

Qiang Ma, Suwen Ge, Danyang He, Darshan Thaker, Iddo Drori

In AAAI Workshop on Deep Learning on Graphs: Methodologies and Applications, 2020. [Arxiv](https://arxiv.org/abs/1911.04936)

## Dependencies
Python>=3.6

PyTorch=1.1


## Baselines
Code for running baselines.


## Small-Scale TSP 
Code, data, and model for small-scale travelling salesman problem (TSP). To train the model, please run train.py via
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

## Larger-Scale TSP
We train the model with small instances and use the model to predict the routes for larger scale TSP, i.e. TSP250/500. Please run the ipython notebook.


## TSPTW
In this experiment, we use hierarchical reinforcement learning to tackle TSP with Time Window (TSPTW). To train hierarchical model, please first train the lower model by
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

## Citation
If you use this code for your research, please cite our papers.
```
@InProceedings{ma2019combinatorial,
  author    = {Ma, Qiang and Ge, Suwen and He, Danyang and Thaker, Darshan and Drori, Iddo},
  title     = {Combinatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning},
  booktitle = {AAAI Workshop on Deep Learning on Graphs: Methodologies and Applications},
  year      = {2020},
}
```
# Final Project - Glassman Yair, Vanelli Martina
## CS236605: Deep Learning 

Our project is based on the article [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984.pdf) (Code: [code](https://github.com/danielzuegner/nettack)). In this paper, the authors generated adversarial attacks on attributed graphs targeting the node’s features and the graph structure. The attacks aim to fool  graph-based deep learning models on a semi-supervised node classification task. 

The main point of our project is to study the behavior of the model and the effectiveness of the adversarial attacks according to the edge densities between and within communities. We studied this problem through Stochastic Block Models. 

This implementation is written in Python 3 and uses Tensorflow for the GCN learning.

## Installation
`conda env create -f environment.yml`
`conda activate project`

## Requirements
* `numpy`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `tensorflow`
* `numba`


## Run the code
 
 To try our code, you can use the IPython notebooks `Par1_Article_Impkementation.ipynb` and `Part2_Article_Implementation.ipynb`.
 
 To run the experiments, one can simply run `python main_experiments.py`. In `main_experiments.py` is also possible to set the main parameters for the experiments.
 
## Results
The results of the experiments can be found in the results folder. The detailed result of a single experiment can be found and repeat through the IPython notebooks.

## References
### Adversarial Attacks on Neural Networks for Graph Data

Implementation of the method proposed in the paper:   
**[Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984)** (Based on the code: [code](https://github.com/danielzuegner/nettack))

by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann.   
Published at SIGKDD'18, August 2018, London, UK

Copyright (C) 2018   
Daniel Zügner   
Technical University of Munich    

[Poster & Presentation Slides](https://www.kdd.in.tum.de/nettack)



### Datasets
In the `data` folder we provide the following datasets originally published by   
#### Cora
McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie.  
*Automating the construction of internet portals with machine learning.*   
Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of   
attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

#### Citeseer
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.

Detailed information about the 2 datasets can be found in the `cora` and `citeseer` folders and in our project report.
### Graph Convolutional Networks
Our implementation of the GCN algorithm is based on the authors' implementation,
available on GitHub [here](https://github.com/tkipf/gcn).

The paper was published as  

Thomas N Kipf and Max Welling. 2017.  
*Semi-supervised classification with graph
convolutional networks.* ICLR (2017).

### SBM
Detailed definitions and SBM analysis can be found in the following article:

Emmanuel Abbe; 18(177):1−86, 2018.  
*Community Detection and Stochastic Block Models: Recent Developments.*  JMLR 2018.
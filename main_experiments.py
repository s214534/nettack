from nettack import utils, GCN
from nettack import nettack as ntk
import numpy as np
from function_plot import plot
import random
from nettack import sbm
import scipy.sparse as sp
import experiment
import warnings

warnings.filterwarnings("ignore")
gpu_id = None # set this to your desired GPU ID if you want to use GPU computations (only for the GCN/surrogate training)
#One can also repeat the experiment with the dataset cora.

#datasets=['citeseer', 'cora']
datasets=['cora']
#community=range(num_communities)
strong=True
n=2
for dataset in datasets:
    exp=experiment.Experiment(dataset, gpu_id=gpu_id)
    exp.compute_p_hat()
    #communities=[]
    #for i in range(exp._K):
    #    communities.append([i])
    communities=[]
    if dataset is 'cora':
        #communities.append(list([2]))
        communities.append(list([4]))
    else:
        communities.append(list([0]))
        communities.append(list([1]))
        communities.append(list([3]))
        
    for community in communities:
        for strong in [True, False]:
            for n in range(1):
                n=0

                exp.compute_p_hat_2(n+1, strong, community)
                exp.generate_SBM()
                exp.split_dataset()
                exp.find_vertices_to_attack()
                exp.pre_run()
                #exp.run()
                #exp.print_file(dataset+"_n"+str(n+1)+"_community"+str(community)+"_strong"+str(strong)+"_factor"+str(2))
                for factor in [0.5, 1, 2]:
                    exp.run(factor=factor)

                    exp.print_file(dataset+"_n"+str(n+1)+"_community"+str(community)+"_strong"+str(strong)+"_factor"+str(factor))
                if community[0] == 0 and strong :
                    n = 2
               
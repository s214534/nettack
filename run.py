from nettack import utils, GCN
from nettack import nettack as ntk
import numpy as np
from function_plot import plot
import random
from nettack import sbm
import scipy.sparse as sp

class run:
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, params_dict={'dropout': 0.5}, gpu_id=0,
                 seed=-1):
        self.
    
    def run_experiment(self, vname, to_namespace=None):
        
        
        
    def surrogate_model()
        surrogate_model_1 = GCN.GCN(sizes, _An_1, _X_obs_hat, with_relu=False, name="surrogate", gpu_id=gpu_id)
        surrogate_model_1.train(split_train, split_val, _Z_obs_hat, print_info=False)
    W1_1 =surrogate_model_1.W1.eval(session=surrogate_model_1.session)
W2_1 =surrogate_model_1.W2.eval(session=surrogate_model_1.session)
    
#Train GCN without perturbations
gcn_before_1 = GCN.GCN(sizes, _An_1, _X_obs_hat, "gcn_orig", gpu_id=gpu_id)
gcn_before_1.train(split_train, split_val, _Z_obs_hat, print_info=False)


#surrogate model of SBM 2 - needed for nettack
surrogate_model_2 = GCN.GCN(sizes, _An_2, _X_obs_hat, with_relu=False, name="surrogate", gpu_id=gpu_id)
surrogate_model_2.train(split_train, split_val, _Z_obs_hat, print_info=False)
W1_2 =surrogate_model_2.W1.eval(session=surrogate_model_2.session)
W2_2 =surrogate_model_2.W2.eval(session=surrogate_model_2.session)
    
#Train GCN without perturbations
gcn_before_2 = GCN.GCN(sizes, _An_2, _X_obs_hat, "gcn_orig", gpu_id=gpu_id)
gcn_before_2.train(split_train, split_val, _Z_obs_hat, print_info=False)



correct=0
mod_correct=0
mod_mis=0
mis=0
succ_att=0
mod_succ_att=0
i=0
limit=500


for u in attacked_nodes:  
    if i ==limit:
        break
    i+=1
    
    ###### AVERAGE COMMUNITIES ########
    n_perturbations=int(_degrees_hat[u]/2)

    clean_1 = gcn_before_1.predictions.eval(session=gcn_before_1.session,feed_dict={gcn_before_1.node_ids: [u]})[0]
       #[clean_1, attacked_1]=test_nettack(_A_obs_hat, _X_obs_hat, _z_obs_hat, u,  n_perturbations, perturb_structure, perturb_features, direct_attack, n_influencers, 
    #         split_train, split_val, split_unlabeled, verbose=False, plot=False)
    
    #if(np.argmax(np.sum(clean_1,axis=1))==_z_obs_hat[u]):
    if(np.argmax(clean_1)==_z_obs_hat[u]):
        correct+=1
        print("Iteration:",i,"/", tot)
        #Set up nettack 
        #NOTE: W1_1 and W_2 are the same for all the attacks and are computed in the previous step
        nettack_1 = ntk.Nettack(_A_obs_hat, _X_obs_hat, _z_obs_hat, W1_1, W2_1, u,verbose=False)
        #Poison the data
        nettack_1.reset()
        nettack_1.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, 
                               perturb_features=perturb_features, direct=direct_attack, n_influencers=n_influencers)
        gcn_retrain_1 = GCN.GCN(sizes, nettack_1.adj_preprocessed, nettack_1.X_obs.tocsr(), 
                                "gcn_retrain", gpu_id=gpu_id)
        gcn_retrain_1.train(split_train, split_val, _Z_obs_hat, print_info=False)
        attacked_1 = gcn_retrain_1.predictions.eval(session=gcn_retrain_1.session,
                                                    feed_dict={gcn_retrain_1.node_ids: [nettack_1.u]})[0]
        #if(np.argmax(np.sum(attacked_1,axis=1))!=_z_obs_hat[u]):
        if(np.argmax(attacked_1)!=_z_obs_hat[u]):
            succ_att+=1
    else:
        mis+=1
        
            
    ######## MODIFIED COMMUNITIES ##########
    n_perturbations=int(_degrees_hat_2[u]/2)
    clean_2 = gcn_before_2.predictions.eval(session=gcn_before_2.session,feed_dict={gcn_before_2.node_ids: [u]})[0]
    
    #if(np.argmax(np.sum(clean_2,axis=1))==_z_obs_hat[u]):
    if(np.argmax(clean_2)==_z_obs_hat[u]):
        mod_correct+=1
        
        #Set up nettack 
        #NOTE: W1_1 and W_2 are the same for all the attacks and are computed in the previous step
        nettack_2 = ntk.Nettack(_A_obs_hat_2, _X_obs_hat, _z_obs_hat, W1_2, W2_2, u, verbose=False)
        #Poison the data
        nettack_2.reset()
        nettack_2.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, 
                                   perturb_features=perturb_features, direct=direct_attack, 
                                   n_influencers=n_influencers)
    
        gcn_retrain_2 = GCN.GCN(sizes, nettack_2.adj_preprocessed, nettack_2.X_obs.tocsr(), "gcn_retrain", 
                                gpu_id=gpu_id)
        gcn_retrain_2.train(split_train, split_val, _Z_obs_hat, print_info=False)
        attacked_2 = gcn_retrain_2.predictions.eval(session=gcn_retrain_2.session,
                                                feed_dict={gcn_retrain_2.node_ids: [nettack_2.u]})[0]
        #if(np.argmax(np.sum(attacked_2,axis=1))!=_z_obs_hat[u]):
        if(np.argmax(attacked_2)!=_z_obs_hat[u]):
            mod_succ_att+=1
    else:
        mod_mis+=1
    
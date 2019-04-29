from matplotlib import pyplot as plt
from nettack import utils, GCN
from nettack import nettack as ntk
import numpy as np
import random
from nettack import sbm
import scipy.sparse as sp

def test_nettack(_A_obs, _X_obs, _z_obs, u,  n_perturbations, perturb_structure, perturb_features, direct_attack, n_influencers, split_train, split_val, split_unlabeled, verbose=True, gpu_id = None, plot=True):
    
    #DATA PREPROCESSING
    _N = _A_obs.shape[0]
    _K = _z_obs.max()+1
    _Z_obs = np.eye(_K)[_z_obs]
    _An = utils.preprocess_graph(_A_obs)
    sizes = [16, _K]
    degrees = _A_obs.sum(0).A1
    neighbors=[]
    for i in range(_A_obs.shape[0]):
        neighbors.append(_A_obs[i].nonzero()[1])
    
    #Train surrogate model (i.e. GCN without nonlinear activation)
    surrogate_model = GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs, print_info=verbose)
    W1 =surrogate_model.W1.eval(session=surrogate_model.session)
    W2 =surrogate_model.W2.eval(session=surrogate_model.session)
    
    #Setup Nettack
    nettack = ntk.Nettack(_A_obs, _X_obs, _z_obs, W1, W2, u, verbose)
    
    #Poison the data
    nettack.reset()
    nettack.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, perturb_features=perturb_features, direct=direct_attack, n_influencers=n_influencers)
    
    #Print perturbations
    if verbose:
        print("Structure perturbations:",nettack.structure_perturbations)
        print("Features perturbations:", nettack.feature_perturbations)
    
    #Train GCN without perturbations
    if verbose:
        print("Train GCN without perturbations")
    retrain_iters=5
    classification_margins_clean = []
    class_distrs_clean = []
    gcn_before = GCN.GCN(sizes, _An, _X_obs, "gcn_orig", gpu_id=gpu_id)
    for _ in range(retrain_iters):
        if verbose:
            print("... {}/{} ".format(_+1, retrain_iters))
        gcn_before.train(split_train, split_val, _Z_obs, print_info=verbose)
        probs_before_attack = gcn_before.predictions.eval(session=gcn_before.session,feed_dict={gcn_before.node_ids: [nettack.u]})[0]
        class_distrs_clean.append(probs_before_attack)
        best_second_class_before = (probs_before_attack - 1000*_Z_obs[nettack.u]).argmax()
        margin_before = probs_before_attack[_z_obs[nettack.u]] - probs_before_attack[best_second_class_before]
        classification_margins_clean.append(margin_before)
    class_distrs_clean = np.array(class_distrs_clean)

    #Train GCN with perturbations
    if verbose:
        print("Train GCN with perturbations")
    classification_margins_corrupted = []
    class_distrs_retrain = []
    gcn_retrain = GCN.GCN(sizes, nettack.adj_preprocessed, nettack.X_obs.tocsr(), "gcn_retrain", gpu_id=gpu_id)
    for _ in range(retrain_iters):
        if verbose:    
            print("... {}/{} ".format(_+1, retrain_iters))
        gcn_retrain.train(split_train, split_val, _Z_obs, print_info=verbose)
        probs_after_attack = gcn_retrain.predictions.eval(session=gcn_retrain.session,feed_dict={gcn_retrain.node_ids: [nettack.u]})[0]
        best_second_class_after = (probs_after_attack - 1000*_Z_obs[nettack.u]).argmax()
        margin_after = probs_after_attack[_z_obs[nettack.u]] - probs_after_attack[best_second_class_after]
        class_distrs_retrain.append(probs_after_attack)
        classification_margins_corrupted.append(margin_after)
    class_distrs_retrain = np.array(class_distrs_retrain)
    
    #Vizualize results
    if plot:
        figure = plt.figure(figsize=(12,4))
        plt.subplot(1, 2, 1)
        center_ixs_clean = []
        for ix, block in enumerate(class_distrs_clean.T):
            x_ixs= np.arange(len(block)) + ix*(len(block)+2)
            center_ixs_clean.append(np.mean(x_ixs))
            color = '#555555'
            if ix == nettack.label_u:
                color = 'darkgreen'
            plt.bar(x_ixs, block, color=color)

        ax=plt.gca()
        plt.ylim((-.05, 1.05))
        plt.ylabel("Predicted probability")
        ax.set_xticks(center_ixs_clean)
        ax.set_xticklabels([make_xlabel(k, nettack.label_u) for k in range(_K)])
        ax.set_title("Predicted class probabilities for node {} on clean data\n({} re-trainings)".format(nettack.u, retrain_iters))

        fig = plt.subplot(1, 2, 2)
        center_ixs_retrain = []
        for ix, block in enumerate(class_distrs_retrain.T):
            x_ixs= np.arange(len(block)) + ix*(len(block)+2)
            center_ixs_retrain.append(np.mean(x_ixs))
            color = '#555555'
            if ix == nettack.label_u:
                color = 'darkgreen'
            plt.bar(x_ixs, block, color=color)


        ax=plt.gca()
        plt.ylim((-.05, 1.05))
        ax.set_xticks(center_ixs_retrain)
        ax.set_xticklabels([make_xlabel(k, nettack.label_u) for k in range(_K)])
        ax.set_title("Predicted class probabilities for node {} after {} perturbations\n({} re-trainings)".format(nettack.u, n_perturbations, retrain_iters))
        plt.tight_layout()
        plt.show()
    
    return class_distrs_clean.T, class_distrs_retrain.T



def make_xlabel(ix, correct):
    if ix==correct:
        return "Class {}\n(correct)".format(ix)
    return "Class {}".format(ix)
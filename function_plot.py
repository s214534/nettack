from matplotlib import pyplot as plt
import numpy as np

def plot(class_distrs_clean, class_distrs_retrain, retrain_iters, u, label_u, _K, n_perturbations):
    figure = plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    center_ixs_clean = []
    for ix, block in enumerate(class_distrs_clean.T):
        x_ixs= np.arange(len(block)) + ix*(len(block)+2)
        center_ixs_clean.append(np.mean(x_ixs))
        color = '#555555'
        if ix == label_u:
            color = 'darkgreen'
        plt.bar(x_ixs, block, color=color)

    ax=plt.gca()
    plt.ylim((-.05, 1.05))
    plt.ylabel("Predicted probability")
    ax.set_xticks(center_ixs_clean)
    ax.set_xticklabels([make_xlabel(k, label_u) for k in range(_K)])
    ax.set_title("Predicted class probabilities for node {} on clean data\n({} re-trainings)".format(u, retrain_iters))

    fig = plt.subplot(1, 2, 2)
    center_ixs_retrain = []
    for ix, block in enumerate(class_distrs_retrain.T):
        x_ixs= np.arange(len(block)) + ix*(len(block)+2)
        center_ixs_retrain.append(np.mean(x_ixs))
        color = '#555555'
        if ix == label_u:
            color = 'darkgreen'
        plt.bar(x_ixs, block, color=color)


    ax=plt.gca()
    plt.ylim((-.05, 1.05))
    ax.set_xticks(center_ixs_retrain)
    ax.set_xticklabels([make_xlabel(k, label_u) for k in range(_K)])
    ax.set_title("Predicted class probabilities for node {} after {} perturbations\n({} re-trainings)".format(u, n_perturbations, retrain_iters))
    plt.tight_layout()
    plt.show()
    

def make_xlabel(ix, correct):
    if ix==correct:
        return "Class {}\n(correct)".format(ix)
    return "Class {}".format(ix)
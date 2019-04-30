from nettack import utils, GCN
from nettack import nettack as ntk
import numpy as np
from function_plot import plot
import random
from nettack import sbm
import scipy.sparse as sp

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


class Experiment:
    def __init__(self, dataset, gpu_id=0,
                 seed=-1):
        self.dataset=dataset
        self.gpu_id=gpu_id 
        self.A_obs, self.X_obs, self.z_obs = utils.load_npz('data/'+dataset+'.npz')
        # in this way we check that N is computed one after the preprocessing
        self.N = self.preprocessing()
        #num of communities
        self._K = self.z_obs.max()+1
        self.num_communities=self._K
        
        
    def preprocessing(self):
        #make the graph indirected, that is make the adjacency matrix _A_obs symmetric
        self.A_obs = self.A_obs + self.A_obs.T 
        self.A_obs[self.A_obs > 1] = 1
        #select the largest connected component    
        lcc = utils.largest_connected_components(self.A_obs)
        self.A_obs = self.A_obs[lcc][:,lcc]
        self.X_obs = self.X_obs[lcc].astype('float32')
        self.z_obs = self.z_obs[lcc]
        #avoid self loops
        self.A_obs=self.A_obs-sp.eye(self.A_obs.shape[0])
        self.A_obs[self.A_obs < 0] = 0
        assert np.abs(self.A_obs - self.A_obs.T).sum() == 0, "Input graph is not symmetric"
        assert self.A_obs.max() == 1 and len(np.unique(self.A_obs[self.A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert self.A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"
        return self.A_obs.shape[0]
    
    def compute_p_hat(self):
        vertices=np.array(range(self.N))
        #vector that stores the # of nodes for each community
        dim_communities=np.zeros(self.num_communities)
        #computes the values and fills the vector
        for i in range(self.num_communities):
            dim_communities[i]=len(vertices[self.z_obs==i])

        #upper triangular matrix of _A_obs,used to compute the ammount of links among members of 
        #the same community
        temp=sp.triu(self.A_obs,k=0).todense()
    
        p_hat=np.zeros([self.num_communities,self.num_communities])
        #for each community
        for i in range(self.num_communities):
            #total number of edges that link members of community i 
            p_hat[i][i]=temp[self.z_obs==i][:,self.z_obs==i].sum()
            #this value has to be normalized over the total number of possible edges that link
            #members of the same community, that is dim_i(dim_i-1)/2= (dim_i)(dim_i-1)/2
            p_hat[i][i]=2*p_hat[i][i]/(dim_communities[i]*(dim_communities[i]-1))
            #for all the other communities
            for j in range(i+1, self.num_communities):
                #total number of edges that link a member of community i and a member of community j
                p_hat[i][j]=self.A_obs[self.z_obs==i][:,self.z_obs==j].sum()
                #over all the possible links, that is dim_i*dim_j
                p_hat[i][j]=p_hat[i][j]/(dim_communities[i]*dim_communities[j])
                p_hat[j][i]=p_hat[i][j]
            
        self.p_hat=p_hat
        
        
    def compute_p_hat_2(self, n, strong, community):
        p_hat_2=np.copy(self.p_hat)
        for i in community:
            #n=sum(_z_obs==i)/4
            if strong:
                p_hat_2[i][i]=p_hat_2[i][i]+n/(sum(self.z_obs==i)-1)
                sum_not_i=sum(self.z_obs!=i)
                for j in range(self.num_communities):
                    if j!=i:
                        p_hat_2[i][j]=p_hat_2[i][j]-n/sum_not_i
                        if p_hat_2[i][j]<0:
                            p_hat_2[i][j]=0
                        p_hat_2[j][i]=p_hat_2[i][j]
            else:
                p_hat_2[i][i]=p_hat_2[i][i]-n/(sum(self.z_obs==i)-1)
                if p_hat_2[i][i]<0:
                    p_hat_2[i][i]=0
                sum_not_i=sum(self.z_obs!=i)
                for j in range(self.num_communities):
                    if j!=i:
                        p_hat_2[i][j]=p_hat_2[i][j]+n/sum_not_i
                        p_hat_2[j][i]=p_hat_2[i][j]

        self.n=n
        self.p_hat_2=p_hat_2
        self.strong=strong
        self.community=community
        
        
    def generate_SBM(self):
        rg=sbm.SBM(self.N, self._K, self.z_obs, self.p_hat, self.p_hat_2)
        self.A_obs_hat=sp.csr_matrix(rg.block_matrix)
        self.A_obs_hat_2=sp.csr_matrix(rg.block_matrix_2)
        self._N = self.preprocessing2()
        
    
    def preprocessing2(self):
        #we select just the largest connected component    
        lcc_1= utils.largest_connected_components(self.A_obs_hat)
        lcc_2= utils.largest_connected_components(self.A_obs_hat_2)
        lcc=intersection(lcc_1,lcc_2)
        self._A_obs_hat = self.A_obs_hat[lcc][:,lcc]
        self._A_obs_hat_2 = self.A_obs_hat_2[lcc][:,lcc]
        self._X_obs_hat = self.X_obs[lcc].astype('float32')
        self._z_obs_hat = self.z_obs[lcc]
        return  self._A_obs_hat.shape[0]
    
    def split_dataset(self, seed=15, unlabeled_share = 0.8, val_share = 0.1 ):
        train_share = 1 - unlabeled_share - val_share
        np.random.seed(seed)
        self.split_train, self.split_val, self.split_unlabeled = utils.train_val_test_split_tabular(np.arange(self._N),
                                                                       train_size=train_share,
                                                                       val_size=val_share,
                                                                       test_size=unlabeled_share,
                                                                       stratify=self._z_obs_hat)
        
    def find_vertices_to_attack(self):
        #degrees and neighbors
        _N=self._N
        neighbors_hat=[]
        _degrees_hat=np.zeros(_N)
        for i in range(_N):
            neighbors_hat.append(self._A_obs_hat[i].nonzero()[1])
            _degrees_hat[i]=len(neighbors_hat[i])
            
        #degrees and neighbors
        neighbors_hat_2=[]
        _degrees_hat_2=np.zeros(_N)
        for i in range(_N):
            neighbors_hat_2.append(self._A_obs_hat_2[i].nonzero()[1])
            _degrees_hat_2[i]=len(neighbors_hat_2[i])

        _vertices=np.array(range(_N))
        indices= self._z_obs_hat==self.community[0]
        self.attacked_nodes = _vertices[[indices[i]  and _degrees_hat[i]>1 and _degrees_hat_2[i]>1 for i in range(_N)]]
        self.tot=len(self.attacked_nodes)
        self._degrees_hat_2=_degrees_hat_2
        self._degrees_hat=_degrees_hat
        
    def pre_run(self):
        self._Z_obs_hat = np.eye(self._K)[self._z_obs_hat]
        self.sizes = [16, self._K]
        _An_1 = utils.preprocess_graph(self._A_obs_hat)
        _An_2 = utils.preprocess_graph(self._A_obs_hat_2)
        surrogate_model_1 = GCN.GCN(self.sizes, _An_1, self._X_obs_hat, with_relu=False, name="surrogate", gpu_id=self.gpu_id)
        surrogate_model_1.train(self.split_train, self.split_val, self._Z_obs_hat, print_info=False)
        self.W1_1 =surrogate_model_1.W1.eval(session=surrogate_model_1.session)
        self.W2_1 =surrogate_model_1.W2.eval(session=surrogate_model_1.session)
        #self.surrogate_model_1=surrogate_model_1
        #Train GCN without perturbations
        self.gcn_before_1 = GCN.GCN(self.sizes, _An_1, self._X_obs_hat, "gcn_orig", gpu_id=self.gpu_id)
        self.gcn_before_1.train(self.split_train, self.split_val, self._Z_obs_hat, print_info=False)


        #surrogate model of SBM 2 - needed for nettack
        surrogate_model_2 = GCN.GCN(self.sizes, _An_2, self._X_obs_hat, with_relu=False, name="surrogate", gpu_id=self.gpu_id)
        surrogate_model_2.train(self.split_train, self.split_val, self._Z_obs_hat, print_info=False)
        self.W1_2 =surrogate_model_2.W1.eval(session=surrogate_model_2.session)
        self.W2_2 =surrogate_model_2.W2.eval(session=surrogate_model_2.session)
    
        #Train GCN without perturbations
        self.gcn_before_2 = GCN.GCN(self.sizes, _An_2, self._X_obs_hat, "gcn_orig", gpu_id=self.gpu_id)
        self.gcn_before_2.train(self.split_train, self.split_val, self._Z_obs_hat, print_info=False)

        
    def run(self, factor=2, limit=250, direct_attack = True, n_influencers = 1, perturb_features = False, perturb_structure = True,verbose = False):
        self.factor=factor
        self.correct=0
        self.mis=0
        self.mod_correct=0
        self.mod_mis=0
        
        self.succ_att=0
        self.mod_succ_att=0
        i=0
        self.tot = min(self.tot, limit)
        for u in self.attacked_nodes:
            if i ==limit:
                break
            if i%5 == 0:
                print("Iteration ", i, "/", self.tot)
            i+=1
    
            ###### AVERAGE COMMUNITIES ########
            n_perturbations=int(self._degrees_hat[u]/factor)

            clean_1 = self.gcn_before_1.predictions.eval(session=self.gcn_before_1.session,feed_dict={self.gcn_before_1.node_ids: [u]})[0]
      
            if(np.argmax(clean_1)==self._z_obs_hat[u]):
                self.correct+=1
            
                nettack_1 = ntk.Nettack(self._A_obs_hat, self._X_obs_hat, self._z_obs_hat, self.W1_1, self.W2_1,            u,verbose=False)
            #Poison the data
                nettack_1.reset()
                nettack_1.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, 
                               perturb_features=perturb_features, direct=direct_attack, n_influencers=n_influencers)
                gcn_retrain_1 = GCN.GCN(self.sizes, nettack_1.adj_preprocessed, nettack_1.X_obs.tocsr(), 
                                "gcn_retrain", gpu_id=self.gpu_id)
                gcn_retrain_1.train(self.split_train, self.split_val, self._Z_obs_hat, print_info=False)
                attacked_1 = gcn_retrain_1.predictions.eval(session=gcn_retrain_1.session,
                                                    feed_dict={gcn_retrain_1.node_ids: [nettack_1.u]})[0]
        
                if(np.argmax(attacked_1)!=self._z_obs_hat[u]):
                    self.succ_att+=1
            else:
                self.mis+=1
        
            
            ######## MODIFIED COMMUNITIES ##########
            n_perturbations=int(self._degrees_hat_2[u]/factor)
            clean_2 =self.gcn_before_2.predictions.eval(session=self.gcn_before_2.session,feed_dict={self.gcn_before_2.node_ids: [u]})[0]
    
    
            if(np.argmax(clean_2)==self._z_obs_hat[u]):
                self.mod_correct+=1
        
                #Set up nettack 
                #NOTE: W1_1 and W_2 are the same for all the attacks and are computed in the previous step
                nettack_2 = ntk.Nettack(self._A_obs_hat_2, self._X_obs_hat, self._z_obs_hat, self.W1_2, self.W2_2, u, verbose=False)
                #Poison the data
                nettack_2.reset()
                nettack_2.attack_surrogate(n_perturbations, perturb_structure=perturb_structure, 
                                   perturb_features=perturb_features, direct=direct_attack, 
                                   n_influencers=n_influencers)
    
                gcn_retrain_2 = GCN.GCN(self.sizes, nettack_2.adj_preprocessed, nettack_2.X_obs.tocsr(), "gcn_retrain", 
                                gpu_id=self.gpu_id)
                gcn_retrain_2.train(self.split_train, self.split_val, self._Z_obs_hat, print_info=False)
                attacked_2 = gcn_retrain_2.predictions.eval(session=gcn_retrain_2.session,
                                                feed_dict={gcn_retrain_2.node_ids: [nettack_2.u]})[0]
        
                if(np.argmax(attacked_2)!=self._z_obs_hat[u]):
                    self.mod_succ_att+=1
            else:
                self.mod_mis+=1


    def print_file(self, name):
        f=open(name,"w+")
        f.write("#### CASE: community="+ str(self.community)+" N="+ str(self.n)+ ", n_perturbations=degree/"+str(self.factor)+ ", strong=" + str(self.strong)+ " on " +self.dataset+" ####\n" )
        f.write("# of tested nodes:"+str(self.tot)+ "\n")    
        f.write("# of correct classified nodes in the first dataset:"+str(self.correct)+ ", accuracy:"+str(self.correct/self.tot)+ "\n")
        f.write("# of correct classified nodes in the second dataset:"+str(self.mod_correct)+ ", accuracy:"+str(self.mod_correct/self.tot)+ "\n")
        if self.correct>0:
            f.write("# of successful attacks in the first dataset:" +str(self.succ_att)+ ", accuracy:"+ str(self.succ_att/self.correct)+ "\n")
        if self.mod_correct>0:
            f.write("# of successful attacks in the second dataset:"+str(self.mod_succ_att)+ ", accuracy:" + str(self.mod_succ_att/self.mod_correct)+ "\n")
        f.close()









    


    











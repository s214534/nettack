import numpy as np
import scipy.sparse as sp
#INPUT: 
#neighbors: list (?) of length N (number of vertices), entry i contains the list/vector/? of the nrighbors of vertex i
#num_communities: integer that represents the number of communities (that is also the number of labels)
#community_labels: for each vertex, his community/label
#norm=True if we want the probabilities (if we want just the number of links between each couple of communities
#     we can set norm to be false)


def avg_matrix(neighbors, num_communities, community_labels, norm=True, ):
    
    #initialization of the matrix avg_p_matrix 
    #each entry of this matrix represents the fraction of the number of links between couples that come 
    #respectively from community i and community j over the sum of all the possibles links that could occur between elements
    #community i and community j
    avg_p_matrix=np.zeros([num_communities,num_communities])
    
    #N=number of vertices
    N=len(neighbors)
    
    #initialize the vector of vertices' indices, that is a vector that goes from 0 to N-1, 
    #each number represents a different vertex
    vertices=np.array(range(0,N))
    
    #initialization of the vector dens_communities
    #for each entry (index that represents a community/label), it memorizes how many vertices there are in that community
    dim_communities=np.zeros(num_communities)
    #computes the values and fills the vector
    for i in range(num_communities):
        dim_communities[i]=len(vertices[community_labels==i])
    
    #computes the percentage of links within each couples of communities over all the possible links within these 2 communities
    for i in range(num_communities):
        #n=vertices[community_labels==i].sum()
        #n=0
        temp=np.copy(community_labels)
        #for each vertex in that community(label)
        for v in vertices[community_labels==i]:
                
                flag=0
                #n=n+1
                temp[vertices<v]=-1
                avg_p_matrix[i,i]+=(temp[neighbors[v]]==i).sum()#/(N-1)
                
                #avg_p_matrix[i,i]+=(community_labels[neighbors[v]]==i).sum()/degrees[v]
                
                #avg_p_matrix_overall[i,i]=(community_labels[neighbors[v]]==i).sum()/N
                    
                for j in range(num_communities):
                    if i!=j:
                        avg_p_matrix[i,j]+=(community_labels[neighbors[v]]==j).sum()#/(N-1)
                        #avg_p_matrix[i,j]+=(community_labels[neighbors[v]]==j).sum()/degrees[v]
                        
                        #if (i==0 and j==1) or (i==1 and j==0):
                            #print(neighbors[v])
                            #print(community_labels[neighbors[v]])
                            #print((community_labels[neighbors[v]]==j).sum()/degrees[v])
                    #flag=0
                #print((community_labels[neighbors[v]]==i).shape[0])
                if flag==1:
                    print(neighbors[v])
                    print(community_labels[neighbors[v]])
                    print((community_labels[neighbors[v]]==i).sum()/N)
                    print((community_labels[neighbors[v]]==i).sum()/degrees[v])
                    flag=0
                
        
        #if i==0:
        #    print(avg_p_matrix[i,i])
        #avg_p_matrix[i,:]=avg_p_matrix[i,:]/n
        for j in range(num_communities):
            avg_p_matrix[i,j]=avg_p_matrix[i,j]/(dim_communities[j]*dim_communities[i])
        #avg_p_matrix_overall[i,i]=avg_p_matrix_overall[i,i]/n
    return avg_p_matrix#, avg_p_matrix_overall
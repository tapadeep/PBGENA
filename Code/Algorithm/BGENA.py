import argparse
import os
import numpy as np
from scipy import sparse
from bitarray import util
import copy
from bitarray import bitarray
from random import random as rando
import pickle
import timeit
class BGENA(object):
    def __init__(self,graph,N,alpha,b_t,b_a,l_t=1,l_a=1,f_t=2,f_a=2):
        print('\nSetting up BGENA...')
        assert os.path.isdir('../../Datasets/'+graph),'Folder for {0} network does not exist'.format(graph)
        self.__graph=graph
        assert os.path.isfile('../../Datasets/'+self.__graph+'/edge_list.npy'),'Edge list file does not exist for {0} network'.format(self.__graph)
        self.__edge_list=np.load('../../Datasets/'+self.__graph+'/edge_list.npy')
        assert os.path.isfile('../../Datasets/'+self.__graph+'/attribute_matrix.npz'),'Attribute matrix file does not exist for {0} network'.format(self.__graph)
        self.__attribute_matrix=sparse.load_npz('../../Datasets/'+self.__graph+'/attribute_matrix.npz')
        assert isinstance(N,int),'Dimensions must be an integer'
        self.__N=N
        assert alpha>=0 and alpha<=1,'alpha should lie in the range [0,1]'
        self.__alpha=alpha
        assert b_t>=0 and b_t<=1,'b_t should lie in the range [0,1]'
        self.__b_t=b_t
        assert b_a>=0 and b_a<=1,'b_a should lie in the range [0,1]'
        self.__b_a=b_a
        self.__N_t=round((1-self.__alpha)*self.__N)
        self.__N_a=self.__N-self.__N_t
        assert isinstance(l_t,int),'Topology level must be an integer'
        self.__l_t=l_t
        assert isinstance(l_a,int),'Attribute level must be an integer'
        self.__l_a=l_a
        assert f_t>=1,'f_t>=1, becuase b_t cannot increase over several passes'
        self.__f_t=f_t
        assert f_a>=1,'f_a>=1, becuase b_a cannot increase over several passes'
        self.__f_a=f_a
        self.__nodes=self.__attribute_matrix.shape[0]
        self.__attributes=self.__attribute_matrix.shape[1]
    def preprocess_edges(self):
        print('\nRemoving unwanted edges...')
        e=set()
        for i in self.__edge_list:
            if i[0]!=i[1] and (i[0],i[1]) not in e and (i[1],i[0]) not in e:
                e.add((i[0],i[1]))
        self.__edge_list=np.zeros((len(e),2),dtype=int)
        j=0
        for i in e:
            self.__edge_list[j]=i
            j+=1
        self.__edges=self.__edge_list.shape[0]
        print('\n{0}:'.format(self.__graph))
        print('#Nodes =',self.__nodes)
        print('#Edges =',self.__edges)
        print('#Attributes =',self.__attributes)
        return self.__edge_list,self.__nodes
    def remove_edges(self,erf):
        print('\nRandomly removing edges...')
        edge_indices=np.arange(self.__edges)
        positive_edge_test=np.random.choice(a=edge_indices,size=int(self.__edges*erf),replace=False)
        self.__edge_list=np.delete(self.__edge_list,positive_edge_test,axis=0)
        self.__edges=self.__edge_list.shape[0]
        return positive_edge_test,edge_indices
    def __mapping(self):
        print('\nMapping...')
        if self.__N_t>0:
            self.__Pi_t=np.random.randint(low=0,high=self.__N_t,size=self.__nodes)
        if self.__N_a>0:
            self.__Pi_a=np.random.randint(low=0,high=self.__N_a,size=self.__attributes)
    def __sketching(self):
        print('\nSketching...')
        if self.__N_t>0:
            self.__S_t=[util.zeros(self.__N_t) for _ in range(self.__nodes)]
            for i in range(self.__nodes):
                self.__S_t[i][self.__Pi_t[i]]=1
            for i in self.__edge_list:
                self.__S_t[i[0]][self.__Pi_t[i[1]]]=1
                self.__S_t[i[1]][self.__Pi_t[i[0]]]=1
        if self.__N_a>0:
            self.__S_a=[util.zeros(self.__N_a) for _ in range(self.__nodes)]
            n,a=self.__attribute_matrix.nonzero()
            for i in range(len(n)):
                self.__S_a[n[i]][self.__Pi_a[a[i]]]=1
    def __edge_propagation(self):
        print('\nEdge Propagation...')
        if self.__N_t>0:
            if self.__b_t==0:
                self.__E_t=self.__S_t
            else:
                for i in range(self.__l_t):
                    self.__E_t=copy.deepcopy(self.__S_t)
                    if self.__b_t==1:
                        for j in self.__edge_list:
                            self.__E_t[j[1]]=self.__S_t[j[0]]|self.__E_t[j[1]]
                            self.__E_t[j[0]]=self.__S_t[j[1]]|self.__E_t[j[0]]
                    else:
                        D_t=[]
                        for j in range(self.__nodes):
                            Q_t=bitarray(rando()<self.__b_t for j in range(self.__N_t))
                            D_t.append(self.__S_t[j]&Q_t)
                        for j in self.__edge_list:
                            self.__E_t[j[1]]=D_t[j[0]]|self.__E_t[j[1]]
                            self.__E_t[j[0]]=D_t[j[1]]|self.__E_t[j[0]]
                        del(D_t,Q_t)
                    self.__S_t=self.__E_t
                    self.__b_t/=self.__f_t
        if self.__N_a>0:
            if self.__b_a==0:
                self.__E_a=self.__S_a
            else:
                for i in range(self.__l_a):
                    self.__E_a=copy.deepcopy(self.__S_a)
                    if self.__b_a==1:
                        for j in self.__edge_list:
                            self.__E_a[j[1]]=self.__S_a[j[0]]|self.__E_a[j[1]]
                            self.__E_a[j[0]]=self.__S_a[j[1]]|self.__E_a[j[0]]
                    else:
                        D_a=[]
                        for j in range(self.__nodes):
                            Q_a=bitarray(rando()<self.__b_a for j in range(self.__N_a))
                            D_a.append(self.__S_a[j]&Q_a)
                        for j in self.__edge_list:
                            self.__E_a[j[1]]=D_a[j[0]]|self.__E_a[j[1]]
                            self.__E_a[j[0]]=D_a[j[1]]|self.__E_a[j[0]]
                        del(D_a,Q_a)
                    self.__S_a=self.__E_a
                    self.__b_a/=self.__f_a
    def embed(self):
        print('\nEmbedding...')
        start_time=timeit.default_timer()
        self.__mapping()
        self.__sketching()
        self.__edge_propagation()
        if self.__N_t==0:
            self.__emb=self.__E_a
        elif self.__N_a==0:
            self.__emb=self.__E_t
        else:
            self.__emb=[self.__E_t[i]+self.__E_a[i] for i in range(self.__nodes)]
        elapsed=timeit.default_timer()-start_time
        print('\nEmbedding Time = %.2fs\n'%elapsed)
        print('Embedding Dimension =',len(self.__emb[0].tolist()),'\n')
        pickle.dump(self.__emb,open('../../Embeddings/'+self.__graph+'_BGENA_emb.pkl','wb'))
        return self.__emb
    def embedding_as_array(self):
        print('Embedding as numpy array...\n')
        for i in range(len(self.__emb)):
            self.__emb[i]=np.frombuffer(self.__emb[i].unpack(),dtype=bool)
        self.__emb=np.array(self.__emb)
        return self.__emb
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='BGENA')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--N',type=int,default=2000,help='Embedding Dimension')
    parser.add_argument('--alpha',type=float,help='Fraction of the dimensions to be used for attributes')
    parser.add_argument('--b_t',type=float,help='Topology Bitset Probability')
    parser.add_argument('--b_a',type=float,help='Attribute Bitset Probability')
    parser.add_argument('--l_t',type=int,default=1,help='Number of passes of edge propagation over the topology embeddings')
    parser.add_argument('--l_a',type=int,default=1,help='Number of passes of edge propagation over the attribute embeddings')
    parser.add_argument('--f_t',type=float,default=2,help='How much to reduce b_t each pass?')
    parser.add_argument('--f_a',type=float,default=2,help='How much to reduce b_a each pass?')
    args=parser.parse_args()
    bgena=BGENA(graph=args.graph,N=args.N,alpha=args.alpha,b_t=args.b_t,b_a=args.b_a,l_t=args.l_t,l_a=args.l_a,f_t=args.f_t,f_a=args.f_a)
    bgena.preprocess_edges()
    bgena.embed()

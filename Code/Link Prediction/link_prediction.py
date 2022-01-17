import sys
sys.path.insert(0,'../Algorithm')
import BGENA
import PBGENA
import argparse
import numpy as np
import random
from scipy import sparse
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import timeit
import os
class LinkPredictor(object):
    def __init__(self,graph,algorithm,erf,**kwargs):
        print('\nSetting up Link Predictor...')
        self.__graph=graph
        self.__algorithm=algorithm
        assert erf>0 and erf<1,'Edge Removed Fraction must be in the range (0,1)'
        self.__erf=erf
        if self.__algorithm=='BGENA':
            self.__N=kwargs['N']
            self.__alpha=kwargs['alpha']
            self.__b_t=kwargs['b_t']
            self.__b_a=kwargs['b_a']
            self.__l_t=kwargs['l_t']
            self.__l_a=kwargs['l_a']
            self.__f_t=kwargs['f_t']
            self.__f_a=kwargs['f_a']
            self.__model=BGENA.BGENA(graph=self.__graph,N=self.__N,alpha=self.__alpha,b_t=self.__b_t,b_a=self.__b_a,l_t=self.__l_t,l_a=self.__l_a,f_t=self.__f_t,f_a=self.__f_a)
        if self.__algorithm=='PBGENA':
            self.__N=kwargs['N']
            self.__alpha=kwargs['alpha']
            self.__b_t=kwargs['b_t']
            self.__b_a=kwargs['b_a']
            self.__p=kwargs['p']
            self.__f=kwargs['f']
            os.chdir('../Algorithm')
            self.__model=PBGENA.PBGENA(graph=self.__graph,p=self.__p,N=self.__N,alpha=self.__alpha,b_t=self.__b_t,b_a=self.__b_a,f=self.__f)
    def generate_negative_edges(self):
        print('\nGenerating negative edges...')
        self.__positive_edge_set,self.__nodes=self.__model.preprocess_edges()
        self.__edges=self.__positive_edge_set.shape[0]
        data=np.ones(len(self.__positive_edge_set))
        real_edges=set()
        for i in range(len(self.__positive_edge_set)):
            real_edges.add((self.__positive_edge_set[i][0],self.__positive_edge_set[i][1]))
        self.__negative_edge_set=np.zeros((self.__edges,2),dtype=int)
        i=0
        while True:
            while i<self.__edges:
                node_1=random.randint(0,self.__nodes-1)
                node_2=random.randint(0,self.__nodes-1)
                if node_1!=node_2 and (node_1,node_2) not in real_edges and (node_2,node_1) not in real_edges:
                    if node_1<node_2:
                        self.__negative_edge_set[i][0]=node_1
                        self.__negative_edge_set[i][1]=node_2
                    else:
                        self.__negative_edge_set[i][0]=node_2
                        self.__negative_edge_set[i][1]=node_1
                    i+=1
            negative_adjacency_list=sparse.csr_matrix((data,(self.__negative_edge_set[:,0],self.__negative_edge_set[:,1])),shape=(self.__nodes,self.__nodes))
            i=negative_adjacency_list.nnz
            if i==self.__edges:
                break
            self.__negative_edge_set[0:i,0]=negative_adjacency_list.nonzero()[0]
            self.__negative_edge_set[0:i,1]=negative_adjacency_list.nonzero()[1]
            i-=1
    def create_residual(self):
        print('\nCreating residual graph...')
        self.__positive_edge_test,edge_indices=self.__model.remove_edges(erf=self.__erf)
        self.__positive_edge_train=np.setdiff1d(edge_indices,self.__positive_edge_test,assume_unique=True)
        self.__negative_edge_test=np.random.choice(a=edge_indices,size=int(self.__edges*self.__erf),replace=False)
        self.__negative_edge_train=np.setdiff1d(edge_indices,self.__negative_edge_test,assume_unique=True)
    def run(self):
        print('\nEmbedding residual graph and forming train-test data...')
        self.__emb=self.__model.embed()
        self.__X_train=np.zeros(len(self.__positive_edge_train)+len(self.__negative_edge_train))
        self.__Y_train=np.zeros(len(self.__positive_edge_train)+len(self.__negative_edge_train))
        j=0
        for i in self.__positive_edge_train:
            self.__X_train[j]=distance.cosine(np.frombuffer(self.__emb[self.__positive_edge_set[i][0]].unpack(),dtype=bool),np.frombuffer(self.__emb[self.__positive_edge_set[i][1]].unpack(),dtype=bool))
            self.__Y_train[j]=1
            j+=1
        for i in self.__negative_edge_train:
            self.__X_train[j]=distance.cosine(np.frombuffer(self.__emb[self.__negative_edge_set[i][0]].unpack(),dtype=bool),np.frombuffer(self.__emb[self.__negative_edge_set[i][1]].unpack(),dtype=bool))
            j+=1
        self.__X_train=np.array(self.__X_train).reshape(-1,1)
        self.__Y_train=np.array(self.__Y_train)
        self.__X_test=np.zeros(len(self.__positive_edge_test)+len(self.__negative_edge_test))
        self.__Y_test=np.zeros(len(self.__positive_edge_test)+len(self.__negative_edge_test))
        j=0
        for i in self.__positive_edge_test:
            self.__X_test[j]=distance.cosine(np.frombuffer(self.__emb[self.__positive_edge_set[i][0]].unpack(),dtype=bool),np.frombuffer(self.__emb[self.__positive_edge_set[i][1]].unpack(),dtype=bool))
            self.__Y_test[j]=1
            j+=1
        for i in self.__negative_edge_test:
            self.__X_test[j]=distance.cosine(np.frombuffer(self.__emb[self.__negative_edge_set[i][0]].unpack(),dtype=bool),np.frombuffer(self.__emb[self.__negative_edge_set[i][1]].unpack(),dtype=bool))
            j+=1
        self.__X_test=np.array(self.__X_test).reshape(-1,1)
        self.__Y_test=np.array(self.__Y_test)
    def predict(self):
        print('Training Predictor...')
        self.__model=LogisticRegression(random_state=0,max_iter=1000)
        self.__model.fit(self.__X_train,self.__Y_train)
        self.__auc_roc=roc_auc_score(self.__Y_test,self.__model.predict_proba(self.__X_test)[:,1])
        self.__ap=average_precision_score(self.__Y_test,self.__model.predict_proba(self.__X_test)[:,1])
        print('\nPerformance:')
        print('Area Under Curve for Receiver Operating Characteristic Curve =',self.__auc_roc)
        print('Average Precision =',self.__ap)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Link Prediction')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--algorithm',type=str,default='PBGENA',help='Algorithm Name')
    parser.add_argument('--erf',type=float,default=0.3,help='Fraction of edges removed')
    parser.add_argument('--N',type=int,help='Embedding Dimension')
    parser.add_argument('--alpha',type=float,default=0.5,help='Fraction of the dimensions to be used for attributes')
    parser.add_argument('--b_t',type=float,default=0.5,help='Topology Bitset Probability')
    parser.add_argument('--b_a',type=float,default=0.5,help='Attribute Bitset Probability')
    parser.add_argument('--l_t',type=int,default=1,help='Number of passes of edge propagation over the topology embeddings')
    parser.add_argument('--l_a',type=int,default=1,help='Number of passes of edge propagation over the attribute embeddings')
    parser.add_argument('--f_t',type=float,default=2,help='How much to reduce b_t each pass?')
    parser.add_argument('--f_a',type=float,default=2,help='How much to reduce b_a each pass?')
    parser.add_argument('--p',type=int,default=16,help='Number of cores')
    parser.add_argument('--f',type=int,default=1,help='Number of fragments')
    args=parser.parse_args()
    link_predictor=LinkPredictor(graph=args.graph,algorithm=args.algorithm,erf=args.erf,N=args.N,alpha=args.alpha,b_t=args.b_t,b_a=args.b_a,l_t=args.l_t,l_a=args.l_a,f_t=args.f_t,f_a=args.f_a,p=args.p,f=args.f)
    start_time=timeit.default_timer()
    link_predictor.generate_negative_edges()
    link_predictor.create_residual()
    link_predictor.run()
    link_predictor.predict()
    elapsed=timeit.default_timer()-start_time
    print('\nTotal Prediction Time = {0}s\n'.format(round(elapsed,4)))
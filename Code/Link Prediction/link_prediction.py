import sys
sys.path.insert(0,'../Algorithm')
import BGENA
import PBGENA
import Baseline
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
        self.__N=kwargs['N']
        self.__alpha=kwargs['alpha']
        self.__b_t=kwargs['b_t']
        self.__b_a=kwargs['b_a']
        self.__l_t=kwargs['l_t']
        self.__l_a=kwargs['l_a']
        self.__f_t=kwargs['f_t']
        self.__f_a=kwargs['f_a']
        self.__p=kwargs['p']
        self.__f=kwargs['f']
        self.__seed=kwargs['seed']
        self.__reduction_dimensions=kwargs['reduction_dimensions']
        self.__svd_iterations=kwargs['svd_iterations']
        self.__theta_max=kwargs['theta_max']
        self.__eval_points=kwargs['eval_points']
        self.__order=kwargs['order']
        self.__walk_number=kwargs['walk_number']
        self.__walk_length=kwargs['walk_length']
        self.__dimensions=kwargs['dimensions']
        self.__workers=kwargs['workers']
        self.__window_size=kwargs['window_size']
        self.__epochs=kwargs['epochs']
        self.__learning_rate=kwargs['learning_rate']
        self.__down_sampling=kwargs['down_sampling']
        self.__min_count=kwargs['min_count']
        self.__alpha=kwargs['alpha']
        self.__iterations=kwargs['iterations']
        self.__binarization_iterations=kwargs['binarization_iterations']
        self.__lower_control=kwargs['lower_control']
        self.__beta=kwargs['beta']
        self.__lambd=kwargs['lambd']
        self.__alpha_1=kwargs['alpha_1']
        self.__alpha_2=kwargs['alpha_2']
        self.__alpha_3=kwargs['alpha_3']
        self.__beta_1=kwargs['beta_1']
        self.__beta_2=kwargs['beta_2']
        self.__beta_3=kwargs['beta_3']
        if self.__algorithm=='BGENA':
            self.__model=BGENA.BGENA(graph=self.__graph,N=self.__N,alpha=self.__alpha,b_t=self.__b_t,b_a=self.__b_a,l_t=self.__l_t,l_a=self.__l_a,f_t=self.__f_t,f_a=self.__f_a)
        elif self.__algorithm=='PBGENA':
            os.chdir('../Algorithm')
            self.__model=PBGENA.PBGENA(graph=self.__graph,p=self.__p,N=self.__N,alpha=self.__alpha,b_t=self.__b_t,b_a=self.__b_a,f=self.__f)
        else:
            self.__model=Baseline.Baseline(graph=self.__graph,algorithm=self.__algorithm,seed=self.__seed,reduction_dimensions=self.__reduction_dimensions,svd_iterations=self.__svd_iterations,theta_max=self.__theta_max,eval_points=self.__eval_points,order=self.__order,walk_number=self.__walk_number,walk_length=self.__walk_length,dimensions=self.__dimensions,workers=self.__workers,window_size=self.__window_size,epochs=self.__epochs,learning_rate=self.__learning_rate,down_sampling=self.__down_sampling,min_count=self.__min_count,alpha=self.__alpha,iterations=self.__iterations,binarization_iterations=self.__binarization_iterations,lower_control=self.__lower_control,beta=self.__beta,lambd=self.__lambd,alpha_1=self.__alpha_1,alpha_2=self.__alpha_2,alpha_3=self.__alpha_3,beta_1=self.__beta_1,beta_2=self.__beta_2,beta_3=self.__beta_3)
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
        print('\nEmbedding residual graph...')
        self.__emb=self.__model.embed()
        if self.__algorithm=='BGENA' or self.__algorithm=='PBGENA':
            self.__emb=self.__model.embedding_as_array()
        print('Generating train-test sets...')
        self.__X_train=np.zeros(len(self.__positive_edge_train)+len(self.__negative_edge_train))
        self.__Y_train=np.zeros(len(self.__positive_edge_train)+len(self.__negative_edge_train))
        j=0
        for i in self.__positive_edge_train:
            self.__X_train[j]=distance.cosine(self.__emb[self.__positive_edge_set[i][0]],self.__emb[self.__positive_edge_set[i][1]])
            self.__Y_train[j]=1
            j+=1
        for i in self.__negative_edge_train:
            self.__X_train[j]=distance.cosine(self.__emb[self.__negative_edge_set[i][0]],self.__emb[self.__negative_edge_set[i][1]])
            j+=1
        self.__X_train=np.array(self.__X_train).reshape(-1,1)
        self.__Y_train=np.array(self.__Y_train)
        self.__X_test=np.zeros(len(self.__positive_edge_test)+len(self.__negative_edge_test))
        self.__Y_test=np.zeros(len(self.__positive_edge_test)+len(self.__negative_edge_test))
        j=0
        for i in self.__positive_edge_test:
            self.__X_test[j]=distance.cosine(self.__emb[self.__positive_edge_set[i][0]],self.__emb[self.__positive_edge_set[i][1]])
            self.__Y_test[j]=1
            j+=1
        for i in self.__negative_edge_test:
            self.__X_test[j]=distance.cosine(self.__emb[self.__negative_edge_set[i][0]],self.__emb[self.__negative_edge_set[i][1]])
            j+=1
        self.__X_test=np.array(self.__X_test).reshape(-1,1)
        self.__Y_test=np.array(self.__Y_test)
    def predict(self):
        print('\nTraining Predictor...')
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
    parser.add_argument('--N',type=int,default=2000,help='Embedding Dimension')
    parser.add_argument('--alpha',type=float,default=0.5,help='Fraction of the dimensions to be used for attributes')
    parser.add_argument('--b_t',type=float,default=0.5,help='Topology Bitset Probability')
    parser.add_argument('--b_a',type=float,default=0.5,help='Attribute Bitset Probability')
    parser.add_argument('--l_t',type=int,default=1,help='Number of passes of edge propagation over the topology embeddings')
    parser.add_argument('--l_a',type=int,default=1,help='Number of passes of edge propagation over the attribute embeddings')
    parser.add_argument('--f_t',type=float,default=2,help='How much to reduce b_t each pass?')
    parser.add_argument('--f_a',type=float,default=2,help='How much to reduce b_a each pass?')
    parser.add_argument('--p',type=int,default=16,help='Number of cores')
    parser.add_argument('--f',type=int,default=1,help='Number of fragments')
    parser.add_argument('--seed',type=int,default=random.randint(0,99),help='Random Seed Value')
    parser.add_argument('--reduction_dimensions',type=int,default=64,help='SVD Reduction Dimensions')
    parser.add_argument('--svd_iterations',type=int,default=20,help='SVD Iteration Count')
    parser.add_argument('--theta_max',type=float,default=2.5,help='Maximal Evaluation Point')
    parser.add_argument('--eval_points',type=int,default=25,help='Number of Characteristic Function Evaluation Points')
    parser.add_argument('--order',type=int,default=5,help='Number of Adjacency Matrix Powers')
    parser.add_argument('--walk_number',type=int,default=5,help='Number of Random Walks')
    parser.add_argument('--walk_length',type=int,default=80,help='Length of Random Walks')
    parser.add_argument('--dimensions',type=int,default=32,help='Dimensionality of Embedding')
    parser.add_argument('--workers',type=int,default=4,help='Number of Cores')
    parser.add_argument('--window_size',type=int,default=3,help='Matrix Power Order')
    parser.add_argument('--epochs',type=int,default=1,help='Number of Epochs')
    parser.add_argument('--learning_rate',type=float,default=0.05,help='HogWild! Learning Rate')
    parser.add_argument('--down_sampling',type=float,default=0.0001,help='Down Sampling Rate in the Corpus')
    parser.add_argument('--min_count',type=int,default=1,help='Minimal Count of Node Occurrences')
    parser.add_argument('--iterations',type=int,default=100,help='Matrix Decomposition Iterations')
    parser.add_argument('--binarization_iterations',type=int,default=20,help='Binarization Iterations')
    parser.add_argument('--lower_control',type=float,default=10**-15,help='Embedding Score Minimal Value')
    parser.add_argument('--beta',type=float,default=0.1,help='Feature Matrix Regularization Coefficient')
    parser.add_argument('--lambd',type=float,default=10.0,help='Regularization Coefficient')
    parser.add_argument('--alpha_1',type=float,default=1000.0,help='Alignment Parameter for Adjacency Matrix')
    parser.add_argument('--alpha_2',type=float,default=1.0,help='Adjacency Basis Regularization')
    parser.add_argument('--alpha_3',type=float,default=1.0,help='Adjacency Features Regularization')
    parser.add_argument('--beta_1',type=float,default=1000.0,help='Alignment Parameter for Feature Matrix')
    parser.add_argument('--beta_2',type=float,default=1.0,help='Attribute Basis Regularization')
    parser.add_argument('--beta_3',type=float,default=1.0,help='Attribute Basis Regularization')
    args=parser.parse_args()
    link_predictor=LinkPredictor(graph=args.graph,algorithm=args.algorithm,erf=args.erf,N=args.N,alpha=args.alpha,b_t=args.b_t,b_a=args.b_a,l_t=args.l_t,l_a=args.l_a,f_t=args.f_t,f_a=args.f_a,p=args.p,f=args.f,seed=args.seed,reduction_dimensions=args.reduction_dimensions,svd_iterations=args.svd_iterations,theta_max=args.theta_max,eval_points=args.eval_points,order=args.order,walk_number=args.walk_number,walk_length=args.walk_length,dimensions=args.dimensions,workers=args.workers,window_size=args.window_size,epochs=args.epochs,learning_rate=args.learning_rate,down_sampling=args.down_sampling,min_count=args.min_count,iterations=args.iterations,binarization_iterations=args.binarization_iterations,lower_control=args.lower_control,beta=args.beta,lambd=args.lambd,alpha_1=args.alpha_1,alpha_2=args.alpha_2,alpha_3=args.alpha_3,beta_1=args.beta_1,beta_2=args.beta_2,beta_3=args.beta_3)
    start_time=timeit.default_timer()
    link_predictor.generate_negative_edges()
    link_predictor.create_residual()
    link_predictor.run()
    link_predictor.predict()
    elapsed=timeit.default_timer()-start_time
    print('\nTotal Prediction Time = {0}s\n'.format(round(elapsed,4)))
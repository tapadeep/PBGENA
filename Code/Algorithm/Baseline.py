import argparse
import random
import os
import numpy as np
from scipy import sparse
import networkx as nx
import timeit
import karateclub
import pickle
class Baseline(object):
    def __init__(self,graph,algorithm,**kwargs):
        print('\nSetting up the baseline...')
        assert os.path.isdir('../../Datasets/'+graph),'Folder for {0} network does not exist'.format(graph)
        self.__graph=graph
        assert algorithm in ['FeatherNode','AE','MUSAE','SINE','BANE','TENE','TADW','FSCNMF','ASNE'],'Not an available ANE solver'
        self.__algorithm=algorithm
        if 'seed' in kwargs:
            assert isinstance(kwargs['seed'],int),'Seed must be an integer'
            self.__seed=kwargs['seed']
        if 'reduction_dimensions' in kwargs:
            assert isinstance(kwargs['reduction_dimensions'],int),'SVD Reduction Dimensions must be an integer'
            self.__reduction_dimensions=kwargs['reduction_dimensions']
        if 'svd_iterations' in kwargs:
            assert isinstance(kwargs['svd_iterations'],int),'SVD Iteration Count must be an integer'
            self.__svd_iterations=kwargs['svd_iterations']
        if 'theta_max' in kwargs:
            assert isinstance(kwargs['theta_max'],float),'Maximal Evaluation Point must be a float'
            self.__theta_max=kwargs['theta_max']
        if 'eval_points' in kwargs:
            assert isinstance(kwargs['eval_points'],int),'Number of Characteristic Function Evaluation Points must be an integer'
            self.__eval_points=kwargs['eval_points']
        if 'order' in kwargs:
            assert isinstance(kwargs['order'],int),'Number of Adjacency Matrix Powers must be an integer'
            self.__order=kwargs['order']
        if 'walk_number' in kwargs:
            assert isinstance(kwargs['walk_number'],int),'Number of Random Walks must be an integer'
            self.__walk_number=kwargs['walk_number']
        if 'walk_length' in kwargs:
            assert isinstance(kwargs['walk_length'],int),'Length of Random Walks must be an integer'
            self.__walk_length=kwargs['walk_length']
        if 'dimensions' in kwargs:
            assert isinstance(kwargs['dimensions'],int),'Dimensionality of Embedding must be an integer'
            self.__dimensions=kwargs['dimensions']
        if 'workers' in kwargs:
            assert isinstance(kwargs['workers'],int),'Number of Cores must be an integer'
            self.__workers=kwargs['workers']
        if 'window_size' in kwargs:
            assert isinstance(kwargs['window_size'],int),'Matrix Power Order must be an integer'
            self.__window_size=kwargs['window_size']
        if 'epochs' in kwargs:
            assert isinstance(kwargs['epochs'],int),'Number of Epochs must be an integer'
            self.__epochs=kwargs['epochs']
        if 'learning_rate' in kwargs:
            assert isinstance(kwargs['learning_rate'],float),'HogWild! Learning Rate must be a float'
            self.__learning_rate=kwargs['learning_rate']
        if 'down_sampling' in kwargs:
            assert isinstance(kwargs['down_sampling'],float),'Down Sampling Rate in the Corpus must be a float'
            self.__down_sampling=kwargs['down_sampling']
        if 'min_count' in kwargs:
            assert isinstance(kwargs['min_count'],int),'Minimal Count of Node Occurrences must be an integer'
            self.__min_count=kwargs['min_count']
        if 'alpha' in kwargs:
            assert isinstance(kwargs['alpha'],float),'Kernel Matrix Inversion Parameter must be a float'
            self.__alpha=kwargs['alpha']
        if 'iterations' in kwargs:
            assert isinstance(kwargs['iterations'],int),'Matrix Decomposition Iterations must be an integer'
            self.__iterations=kwargs['iterations']
        if 'binarization_iterations' in kwargs:
            assert isinstance(kwargs['binarization_iterations'],int),'Binarization Iterations must be an integer'
            self.__binarization_iterations=kwargs['binarization_iterations']
        if 'lower_control' in kwargs:
            assert isinstance(kwargs['lower_control'],float),'Embedding Score Minimal Value must be a float'
            self.__lower_control=kwargs['lower_control']
        if 'beta' in kwargs:
            assert isinstance(kwargs['beta'],float),'Feature Matrix Regularization Coefficient must be a float'
            self.__beta=kwargs['beta']
        if 'lambd' in kwargs:
            assert isinstance(kwargs['lambd'],float),'Regularization Coefficient must be a float'
            self.__lambd=kwargs['lambd']
        if 'alpha_1' in kwargs:
            assert isinstance(kwargs['alpha_1'],float),'Alignment Parameter for Adjacency Matrix must be a float'
            self.__alpha_1=kwargs['alpha_1']
        if 'alpha_2' in kwargs:
            assert isinstance(kwargs['alpha_2'],float),'Adjacency Basis Regularization must be a float'
            self.__alpha_2=kwargs['alpha_2']
        if 'alpha_3' in kwargs:
            assert isinstance(kwargs['alpha_3'],float),'Adjacency Features Regularization must be a float'
            self.__alpha_3=kwargs['alpha_3']
        if 'beta_1' in kwargs:
            assert isinstance(kwargs['beta_1'],float),'Alignment Parameter for Feature Matrix must be a float'
            self.__beta_1=kwargs['beta_1']
        if 'beta_2' in kwargs:
            assert isinstance(kwargs['beta_2'],float),'Attribute Basis Regularization must be a float'
            self.__beta_2=kwargs['beta_2']
        if 'beta_3' in kwargs:
            assert isinstance(kwargs['beta_3'],float),'Attribute Basis Regularization must be a float'
            self.__beta_3=kwargs['beta_3']
        assert os.path.isfile('../../Datasets/'+self.__graph+'/edge_list.npy'),'Edge list file does not exist for {0} network'.format(self.__graph)
        self.__edge_list=np.load('../../Datasets/'+self.__graph+'/edge_list.npy')
        assert os.path.isfile('../../Datasets/'+self.__graph+'/attribute_matrix.npz'),'Attribute matrix file does not exist for {0} network'.format(self.__graph)
        self.__attribute_matrix=sparse.load_npz('../../Datasets/'+self.__graph+'/attribute_matrix.npz').tocoo(copy=False)
        self.__nodes=self.__attribute_matrix.shape[0]
        self.__attributes=self.__attribute_matrix.shape[1]
    def preprocess_edges(self):
        print('\nPreparing the graph...')
        self.__network=nx.Graph()
        for i in range(self.__nodes):
            self.__network.add_node(i)
        for i in range(self.__edge_list.shape[0]):
            self.__network.add_edge(self.__edge_list[i][0],self.__edge_list[i][1])
        self.__network.remove_edges_from(nx.selfloop_edges(self.__network))
        self.__edges=self.__network.number_of_edges()
        print('\n{0}:'.format(self.__graph))
        print('#Nodes =',self.__nodes)
        print('#Edges =',self.__edges)
        print('#Attributes =',self.__attributes)
        self.__edge_list=np.array(self.__network.edges)
        return self.__edge_list,self.__nodes
    def remove_edges(self,erf):
        print('\nRandomly removing edges...')
        edge_indices=np.arange(self.__edges)
        positive_edge_test=np.random.choice(a=edge_indices,size=int(self.__edges*erf),replace=False)
        self.__edge_list=np.delete(self.__edge_list,positive_edge_test,axis=0)
        self.__network=nx.create_empty_copy(self.__network)
        self.__network.add_edges_from(self.__edge_list)
        self.__edges=self.__edge_list.shape[0]
        return positive_edge_test,edge_indices
    def embed(self):
        print('\nEmbedding...')
        if self.__algorithm=='FeatherNode':
            self.__model=karateclub.FeatherNode(reduction_dimensions=self.__reduction_dimensions,svd_iterations=self.__svd_iterations,theta_max=self.__theta_max,eval_points=self.__eval_points,order=self.__order,seed=self.__seed)
        if self.__algorithm=='AE':
            self.__model=karateclub.AE(walk_number=self.__walk_number,walk_length=self.__walk_length,dimensions=self.__dimensions,workers=self.__workers,window_size=self.__window_size,epochs=self.__epochs,learning_rate=self.__learning_rate,down_sampling=self.__down_sampling,min_count=self.__min_count,seed=self.__seed)
        if self.__algorithm=='MUSAE':
            self.__model=karateclub.MUSAE(walk_number=self.__walk_number,walk_length=self.__walk_length,dimensions=self.__dimensions,workers=self.__workers,window_size=self.__window_size,epochs=self.__epochs,learning_rate=self.__learning_rate,down_sampling=self.__down_sampling,min_count=self.__min_count,seed=self.__seed)
        if self.__algorithm=='SINE':
            self.__model=karateclub.SINE(walk_number=self.__walk_number,walk_length=self.__walk_length,dimensions=self.__dimensions,workers=self.__workers,window_size=self.__window_size,epochs=self.__epochs,learning_rate=self.__learning_rate,min_count=self.__min_count,seed=self.__seed)
        if self.__algorithm=='BANE':
            self.__model=karateclub.BANE(dimensions=self.__dimensions,svd_iterations=self.__svd_iterations,alpha=self.__alpha,iterations=self.__iterations,binarization_iterations=self.__binarization_iterations,seed=self.__seed)
        if self.__algorithm=='TENE':
            self.__model=karateclub.TENE(dimensions=self.__dimensions,lower_control=self.__lower_control,alpha=self.__alpha,beta=self.__beta,iterations=self.__iterations,seed=self.__seed)
        if self.__algorithm=='TADW':
            self.__model=karateclub.TADW(dimensions=self.__dimensions,reduction_dimensions=self.__reduction_dimensions,svd_iterations=self.__svd_iterations,alpha=self.__alpha,iterations=self.__iterations,lambd=self.__lambd,seed=self.__seed)
        if self.__algorithm=='FSCNMF':
            self.__model=karateclub.FSCNMF(dimensions=self.__dimensions,lower_control=self.__lower_control,iterations=self.__iterations,alpha_1=self.__alpha_1,alpha_2=self.__alpha_2,alpha_3=self.__alpha_3,beta_1=self.__beta_1,beta_2=self.__beta_2,beta_3=self.__beta_3,seed=self.__seed)
        if self.__algorithm=='ASNE':
            self.__model=karateclub.ASNE(dimensions=self.__dimensions,workers=self.__workers,epochs=self.__epochs,down_sampling=self.__down_sampling,learning_rate=self.__learning_rate,min_count=self.__min_count,seed=self.__seed)
        start_time=timeit.default_timer()
        self.__model.fit(self.__network,self.__attribute_matrix)
        elapsed=timeit.default_timer()-start_time
        print('\nEmbedding Time = %.2fs\n'%elapsed)
        self.__emb=self.__model.get_embedding()
        print('Embedding Dimension =',self.__emb.shape[1],'\n')
        pickle.dump(self.__emb,open('../../Embeddings/'+self.__graph+'_'+self.__algorithm+'_emb.pkl','wb'))
        return self.__emb
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--algorithm',type=str,help='Algorithm Name')
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
    parser.add_argument('--alpha',type=float,default=0.3,help='Kernel Matrix Inversion Parameter')
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
    baseline=Baseline(graph=args.graph,algorithm=args.algorithm,seed=args.seed,reduction_dimensions=args.reduction_dimensions,svd_iterations=args.svd_iterations,theta_max=args.theta_max,eval_points=args.eval_points,order=args.order,walk_number=args.walk_number,walk_length=args.walk_length,dimensions=args.dimensions,workers=args.workers,window_size=args.window_size,epochs=args.epochs,learning_rate=args.learning_rate,down_sampling=args.down_sampling,min_count=args.min_count,alpha=args.alpha,iterations=args.iterations,binarization_iterations=args.binarization_iterations,lower_control=args.lower_control,beta=args.beta,lambd=args.lambd,alpha_1=args.alpha_1,alpha_2=args.alpha_2,alpha_3=args.alpha_3,beta_1=args.beta_1,beta_2=args.beta_2,beta_3=args.beta_3)
    baseline.preprocess_edges()
    baseline.embed()

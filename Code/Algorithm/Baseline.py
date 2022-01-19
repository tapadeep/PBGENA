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
        start_time=timeit.default_timer()
        self.__model.fit(self.__network,self.__attribute_matrix)
        elapsed=timeit.default_timer()-start_time
        print('\nEmbedding Time = {0}s\n'.format(round(elapsed,4)))
        self.__emb=self.__model.get_embedding()
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
    args=parser.parse_args()
    baseline=Baseline(graph=args.graph,algorithm=args.algorithm,seed=args.seed,reduction_dimensions=args.reduction_dimensions,svd_iterations=args.svd_iterations,theta_max=args.theta_max,eval_points=args.eval_points,order=args.order)
    baseline.preprocess_edges()
    baseline.embed()
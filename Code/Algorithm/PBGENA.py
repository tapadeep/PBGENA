import argparse
import os
import numpy as np
from scipy import sparse
import pickle
class PBGENA(object):
    def __init__(self,graph,p,N,alpha,b_t,b_a):
        print('\nSetting up PBGENA...')
        assert os.path.isdir('../../Datasets/'+graph),'Folder for {0} network does not exist'.format(graph)
        self.__graph=graph
        assert os.path.isfile('../../Datasets/'+self.__graph+'/edge_list.npy'),'Edge list file does not exist for {0} network'.format(self.__graph)
        assert os.path.isfile('../../Datasets/'+self.__graph+'/attribute_matrix.npz'),'Attribute matrix file does not exist for {0} network'.format(self.__graph)
        attribute_matrix=sparse.load_npz('../../Datasets/'+self.__graph+'/attribute_matrix.npz')
        self.__nodes=attribute_matrix.shape[0]
        self.__attributes=attribute_matrix.shape[1]
        assert isinstance(N,int),'Dimensions must be an integer'
        self.__N=N
        assert alpha>0 and alpha<1,'alpha should lie in the range (0,1)'
        self.__alpha=alpha
        assert b_t>=0 and b_t<=1,'b_t should lie in the range [0,1]'
        self.__b_t=b_t
        assert b_a>=0 and b_a<=1,'b_a should lie in the range [0,1]'
        self.__b_a=b_a
        assert isinstance(p,int),'Number of processors must be an integer'
        self.__p=p
    def preprocess_edges(self):
        print('\nRemoving unwanted edges...')
        edge_list=np.load('../../Datasets/'+self.__graph+'/edge_list.npy')
        e=set()
        for i in edge_list:
            if i[0]!=i[1] and (i[0],i[1]) not in e and (i[1],i[0]) not in e:
                e.add((i[0],i[1]))
        e=list(e)
        edge_list=np.array(e)
        np.save('../../Datasets/'+self.__graph+'/edge_list_preprocessed.npy',edge_list)
        self.__edges=edge_list.shape[0]
        print('\n{0}:'.format(self.__graph))
        print('#Nodes =',self.__nodes)
        print('#Edges =',self.__edges)
        print('#Attributes =',self.__attributes)
        return edge_list,self.__nodes
    def remove_edges(self,erf):
        print('\nRandomly removing edges...')
        edge_list=np.load('../../Datasets/'+self.__graph+'/edge_list_preprocessed.npy')
        edge_indices=np.arange(self.__edges)
        positive_edge_test=np.random.choice(a=edge_indices,size=int(self.__edges*erf),replace=False)
        edge_list=np.delete(edge_list,positive_edge_test,axis=0)
        np.save('../../Datasets/'+self.__graph+'/edge_list_preprocessed.npy',edge_list)
        return positive_edge_test,edge_indices
    def embed(self):
        print('\nEmbedding...')
        file=open('PBGENA_parameters.txt','w+')
        file.write('graph {0}\n'.format(self.__graph))
        file.write('N {0}\n'.format(self.__N))
        file.write('alpha {0}\n'.format(self.__alpha))
        file.write('b_a {0}\n'.format(self.__b_a))
        file.write('b_t {0}\n'.format(self.__b_t))
        file.write('nodes {0}\n'.format(self.__nodes))
        file.write('edges {0}\n'.format(self.__edges))
        file.write('attributes {0}\n'.format(self.__attributes))
        file.close()
        os.system('mpiexec -n {0} python PBGENA_routine.py'.format(self.__p))
        os.remove('../../Datasets/'+self.__graph+'/edge_list_preprocessed.npy')
        emb=pickle.load(open('../../Embeddings/'+self.__graph+'_PBGENA_emb.pkl','rb'))
        return emb
    def embedding_as_array(self):
        print('\nEmbedding as numpy array...')
        emb=pickle.load(open('../../Embeddings/'+self.__graph+'_PBGENA_emb.pkl','rb'))
        for i in range(len(emb)):
            emb[i]=np.frombuffer(emb[i].unpack(),dtype=bool)
        emb=np.array(emb)
        return emb
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='PBGENA')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--N',type=int,default=2000,help='Embedding Dimension')
    parser.add_argument('--alpha',type=float,help='Fraction of the dimensions to be used for attributes')
    parser.add_argument('--b_t',type=float,help='Topology Bitset Probability')
    parser.add_argument('--b_a',type=float,help='Attribute Bitset Probability')
    parser.add_argument('--p',type=int,help='Number of cores')
    args=parser.parse_args()
    pbgena=PBGENA(graph=args.graph,p=args.p,N=args.N,alpha=args.alpha,b_t=args.b_t,b_a=args.b_a)
    pbgena.preprocess_edges()
    import timeit
    start_time=timeit.default_timer()
    pbgena.embed()
    elapsed=timeit.default_timer()-start_time
    print('\nEmbedding Time + Graph Reading Time = {0}s\n'.format(round(elapsed,4)))
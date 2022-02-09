import argparse
import os
import numpy as np
from scipy import sparse
import pickle
import timeit
class PBGENA(object):
    def __init__(self,graph,p,N,alpha,b_t,b_a,l_t=1,l_a=1,f_t=2,f_a=2,f=1):
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
        assert alpha>=0 and alpha<=1,'alpha should lie in the range [0,1]'
        self.__alpha=alpha
        assert b_t>=0 and b_t<=1,'b_t should lie in the range [0,1]'
        self.__b_t=b_t
        assert b_a>=0 and b_a<=1,'b_a should lie in the range [0,1]'
        self.__b_a=b_a
        assert isinstance(p,int),'Number of processors must be an integer'
        self.__p=p
        assert isinstance(f,int),'Number of fragments must be an integer'
        self.__f=f
        assert isinstance(l_t,int),'Topology level must be an integer'
        self.__l_t=l_t
        assert isinstance(l_a,int),'Attribute level must be an integer'
        self.__l_a=l_a
        assert f_t>=1,'f_t>=1, becuase b_t cannot increase over several passes'
        self.__f_t=f_t
        assert f_a>=1,'f_a>=1, becuase b_a cannot increase over several passes'
        self.__f_a=f_a
    def preprocess_edges(self):
        print('\nRemoving unwanted edges...')
        edge_list=np.load('../../Datasets/'+self.__graph+'/edge_list.npy')
        e=set()
        for i in edge_list:
            if i[0]!=i[1] and (i[0],i[1]) not in e and (i[1],i[0]) not in e:
                e.add((i[0],i[1]))
        edge_list=np.zeros((len(e),2),dtype=int)
        j=0
        for i in e:
            edge_list[j]=i
            j+=1
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
        self.__edges=edge_list.shape[0]
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
        file.write('l_t {0}\n'.format(self.__l_t))
        file.write('l_a {0}\n'.format(self.__l_a))
        file.write('f_t {0}\n'.format(self.__f_t))
        file.write('f_a {0}\n'.format(self.__f_a))
        file.write('fragments {0}\n'.format(self.__f))
        file.write('nodes {0}\n'.format(self.__nodes))
        file.write('edges {0}\n'.format(self.__edges))
        file.write('attributes {0}\n'.format(self.__attributes))
        file.close()
        start_time=timeit.default_timer()
        os.system('mpiexec -n {0} python PBGENA_routine.py'.format(self.__p))
        elapsed=timeit.default_timer()-start_time
        print('Embedding Time + Graph Reading Time = %.2fs\n'%elapsed)
        os.remove('../../Datasets/'+self.__graph+'/edge_list_preprocessed.npy')
        emb=pickle.load(open('../../Embeddings/'+self.__graph+'_PBGENA_emb.pkl','rb'))
        print('Embedding Dimension =',len(emb[0].tolist()),'\n')
        return emb
    def embedding_as_array(self):
        print('Embedding as numpy array...\n')
        emb=pickle.load(open('../../Embeddings/'+self.__graph+'_PBGENA_emb.pkl','rb'))
        for i in range(len(emb)):
            emb[i]=np.frombuffer(emb[i].unpack(),dtype=bool)
        emb=np.array(emb)
        return emb
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='PBGENA')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--N',type=int,default=8000,help='Embedding Dimension')
    parser.add_argument('--alpha',type=float,help='Fraction of the dimensions to be used for attributes')
    parser.add_argument('--b_t',type=float,help='Topology Bitset Probability')
    parser.add_argument('--b_a',type=float,help='Attribute Bitset Probability')
    parser.add_argument('--l_t',type=int,default=1,help='Number of passes of edge propagation over the topology embeddings')
    parser.add_argument('--l_a',type=int,default=1,help='Number of passes of edge propagation over the attribute embeddings')
    parser.add_argument('--f_t',type=float,default=2,help='How much to reduce b_t each pass?')
    parser.add_argument('--f_a',type=float,default=2,help='How much to reduce b_a each pass?')
    parser.add_argument('--p',type=int,default=32,help='Number of Cores')
    parser.add_argument('--f',type=int,default=1,help='Number of Fragments')
    args=parser.parse_args()
    pbgena=PBGENA(graph=args.graph,p=args.p,N=args.N,alpha=args.alpha,b_t=args.b_t,b_a=args.b_a,l_t=args.l_t,l_a=args.l_a,f_t=args.f_t,f_a=args.f_a,f=args.f)
    pbgena.preprocess_edges()
    pbgena.embed()

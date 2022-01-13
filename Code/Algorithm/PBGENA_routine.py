from mpi4py import MPI
import numpy as np
from scipy import sparse
from bitarray import util
from bitarray import bitarray
from random import random as rando
import pickle
import timeit
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
file=open('PBGENA_parameters.txt','r+')
while True:
    line=file.readline()
    if not line:
        break
    argument=line.split(' ')
    if argument[0]=='graph':
        graph=argument[1][:-1]
    elif argument[0]=='N':
        N=int(argument[1][:-1])
    elif argument[0]=='alpha':
        alpha=float(argument[1][:-1])
    elif argument[0]=='b_a':
        b_a=float(argument[1][:-1])
    elif argument[0]=='b_t':
        b_t=float(argument[1][:-1])
    elif argument[0]=='nodes':
        nodes=int(argument[1][:-1])
    elif argument[0]=='edges':
        edges=int(argument[1][:-1])
    elif argument[0]=='attributes':
        attributes=int(argument[1][:-1])
del(argument,line)
file.close()
N_t=int((1-alpha)*N)
N_a=N-N_t
if rank==0:
    start_time=timeit.default_timer()
    partition=np.random.randint(low=0,high=size,size=nodes)
else:
    partition=np.empty(nodes,dtype='int')
comm.Bcast(partition)
partitioned_nodes=np.zeros(nodes+1,dtype=int)
j=0
for i in range(nodes):
    if partition[i]==rank:
        partitioned_nodes[j]=i
        j+=1
partitioned_nodes=partitioned_nodes[0:j]
if rank==0:
    time=timeit.default_timer()-start_time
edge_list=np.load('../../Datasets/'+graph+'/edge_list_preprocessed.npy')
in_edges=np.zeros((edges,2),dtype=int)
cross_edges=np.zeros((edges,2),dtype=int)
node_set=partitioned_nodes.tolist()
node_set=set(node_set)
j=0
k=0
for i in edge_list:
    true_1=i[0] in node_set
    true_2=i[1] in node_set
    if true_1 and true_2:
        in_edges[j]=i
        j+=1
    elif true_1:
        cross_edges[k]=i
        k+=1
    elif true_2:
        cross_edges[k][0]=i[1]
        cross_edges[k][1]=i[0]
        k+=1
del(node_set,edge_list)
in_edges=in_edges[0:j,:]
cross_edges=cross_edges[0:k,:]
attribute_matrix=sparse.load_npz('../../Datasets/'+graph+'/attribute_matrix.npz')
attribute_list=dict()
for i in partitioned_nodes:
    attribute_list[i]=attribute_matrix[i].nonzero()[1]
del(attribute_matrix)
if rank==0:
    start_time=timeit.default_timer()
    Pi_t=np.random.randint(low=0,high=N_t,size=nodes)
    Pi_a=np.random.randint(low=0,high=N_a,size=attributes)
else:
    Pi_t=np.empty(nodes,dtype=int)
    Pi_a=np.empty(attributes,dtype=int)
comm.Bcast(Pi_t)
comm.Bcast(Pi_a)
S_a=dict()
for i in partitioned_nodes:
    S_a[i]=util.zeros(N_a)
    for j in attribute_list[i]:
        S_a[i][Pi_a[j]]=1
del(Pi_a,attribute_list)
S_t=dict()
for i in partitioned_nodes:
    S_t[i]=util.zeros(N_t)
    S_t[i][Pi_t[i]]=1
for i in in_edges:
    S_t[i[0]][Pi_t[i[1]]]=1
    S_t[i[1]][Pi_t[i[0]]]=1
for i in cross_edges:
    S_t[i[0]][Pi_t[i[1]]]=1
del(Pi_t)
E_t=dict()
D_t=dict()
for i in partitioned_nodes:
    E_t[i]=S_t[i]
    Q_t=bitarray(rando()<b_t for _ in range(N_t))
    D_t[i]=S_t[i]&Q_t
del(S_t,Q_t)
E_a=dict()
D_a=dict()
for i in partitioned_nodes:
    E_a[i]=S_a[i]
    Q_a=bitarray(rando()<b_a for _ in range(N_a))
    D_a[i]=S_a[i]&Q_a
del(partitioned_nodes,S_a,Q_a)
for i in in_edges:
    E_t[i[1]]=D_t[i[0]]|E_t[i[1]]
    E_t[i[0]]=D_t[i[1]]|E_t[i[0]]
    E_a[i[1]]=D_a[i[0]]|E_a[i[1]]
    E_a[i[0]]=D_a[i[1]]|E_a[i[0]]
del(in_edges)
batch_D_t=[dict() for _ in range(size)]
batch_D_a=[dict() for _ in range(size)]
for i in cross_edges:
    batch_D_t[partition[i[1]]][i[0]]=D_t[i[0]]
    batch_D_a[partition[i[1]]][i[0]]=D_a[i[0]]
del(partition)
D_t=dict()
D_a=dict()
for i in range(size):
    D_t.update(comm.scatter(batch_D_t,root=i))
    D_a.update(comm.scatter(batch_D_a,root=i))
del(batch_D_t,batch_D_a)
for i in cross_edges:
    E_t[i[0]]=D_t[i[1]]|E_t[i[0]]
    E_a[i[0]]=D_a[i[1]]|E_a[i[0]]
del(cross_edges,D_t,D_a)
E_t=comm.gather(E_t)
E_a=comm.gather(E_a)
if rank==0:
    emb=[[] for _ in range(nodes)]
    for i in E_t:
        for k,l in i.items():
            emb[k]=l
    del(E_t)
    for i in E_a:
        for k,l in i.items():
            emb[k]=emb[k]+l
    del(E_a)
    time+=timeit.default_timer()-start_time
    print('\nEmbedding Time = {0}s'.format(round(time,4)),end='')
    pickle.dump(emb,open('../../Embeddings/'+graph+'_PBGENA_emb.pkl','wb'))
    del(emb)
else:
    del(E_t,E_a)
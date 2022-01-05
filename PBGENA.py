from mpi4py import MPI
import timeit
import pandas as pd
import numpy as np
from scipy import sparse
from bitarray import util
from bitarray import bitarray
import random
from random import random as rando
import pickle
import sys
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
graph='ppi'
if rank==0:
    classes='multi'
N=2000
attribute_percentage=10
attribute_bitset_probability=0.95
topology_bitset_probability=0.5
topology_dimension=int((100-attribute_percentage)/100*N)
attribute_dimension=N-topology_dimension
nn=None
aa=None
if rank==0:
    if classes=='multi':
        labels_read=open('../Datasets/'+graph+'/labels.txt','r+').readlines()
        labels=[set() for i in range(len(labels_read))]
        labeled_nodes=[]
        for line in labels_read:
            s=line.strip().split()
            s=[int(i) for i in s]
            labels[s[0]]=set(s[1:])
            if len(s)>1:
                labeled_nodes.append(s[0])
        pickle.dump(labeled_nodes,open('labeled_nodexx.pkl','wb'))
        del(labeled_nodes)
    else:
        labels_read=pd.read_csv('../Datasets/'+graph+'/labels.txt',header=None,sep='\s+')
        if graph=='tweibo':
            labels=np.zeros(2320895,int)
        else:
            labels=np.zeros(labels_read.shape[0],int)
        for i in range(labels_read.shape[0]):
            labels[labels_read[0][i]]=labels_read[1][i]
    del(labels_read)
    nn=len(labels)
    pickle.dump(labels,open('labelxx.pkl','wb'))
    del(labels)
    att=sparse.load_npz('../Datasets/'+graph+'/attrs.npz')
    aa=att.shape[1]
    del(att)
    start_time=timeit.default_timer()
    partition=np.empty(nn,dtype='int')
    for i in range(nn):
        partition[i]=random.randint(0,size-1)
nn=comm.bcast(nn)
aa=comm.bcast(aa)
if rank!=0:
    partition=np.empty(nn,dtype='int')
comm.Bcast(partition)
partitioned_nodes=set()
for i in range(nn):
    if partition[i]==rank:
        partitioned_nodes.add(i)
if rank==0:
    time=timeit.default_timer()-start_time
edge_list=pd.read_csv('../Datasets/'+graph+'/edgelist.txt',header=None,sep='\s+')
edge_list.drop_duplicates(inplace=True,ignore_index=True)
edge_list=edge_list.to_numpy()
adjacency_list=dict()
for i in partitioned_nodes:
    adjacency_list[i]=set()
for i in range(edge_list.shape[0]):
    if partition[edge_list[i][0]]==rank:
        adjacency_list[edge_list[i][0]].add(edge_list[i][1])
    if partition[edge_list[i][1]]==rank:
        adjacency_list[edge_list[i][1]].add(edge_list[i][0])
del(edge_list)
att=sparse.load_npz('../Datasets/'+graph+'/attrs.npz')
attributes=dict()
for i in partitioned_nodes:
    attributes[i]=att[i].nonzero()[1]
del(att)
if rank==0:
    start_time=timeit.default_timer()
topology_mapping=np.empty(nn,dtype='int')
attribute_mapping=np.empty(aa,dtype='int')
if rank==0:
    topology_mapping=np.random.randint(low=0,high=topology_dimension,size=(nn,))
    attribute_mapping=np.random.randint(low=0,high=attribute_dimension,size=(aa,))
comm.Bcast(topology_mapping)
comm.Bcast(attribute_mapping)
topology_sketches=dict()
for i in partitioned_nodes:
    topology_sketches[i]=util.zeros(topology_dimension)
    topology_sketches[i][topology_mapping[i]]=1
    for j in adjacency_list[i]:
        topology_sketches[i][topology_mapping[j]]=1
del(topology_mapping)
attribute_sketches=dict()
for i in partitioned_nodes:
    attribute_sketches[i]=util.zeros(attribute_dimension)
    for j in attributes[i]:
        attribute_sketches[i][attribute_mapping[j]]=1
del(attribute_mapping)
del(attributes)
topology_embeddings=dict()
R=dict()
for i in partitioned_nodes:
    topology_embeddings[i]=topology_sketches[i]
    Q=bitarray(rando()<topology_bitset_probability for j in range(topology_dimension))
    R[i]=topology_sketches[i]&Q
del(topology_sketches)
give_R=[dict() for i in range(size)]
got_R=dict()
to_get=[]
for i in partitioned_nodes:
    for j in adjacency_list[i]:
        if j in partitioned_nodes:
            topology_embeddings[i]=R[j]|topology_embeddings[i]
        else:
            to_get.append((i,j))
            give_R[partition[j]][i]=R[i]
tagger=1
for i in range(size):
    for j in range(size):
        if rank==i and i!=j:
            comm.send(give_R[j],dest=j,tag=tagger)
        elif rank==j and i!=j:
            got_R.update(comm.recv(source=i,tag=tagger))
        tagger+=1
for i in to_get:
    topology_embeddings[i[0]]=got_R[i[1]]|topology_embeddings[i[0]]
attribute_embeddings=dict()
R=dict()
for i in partitioned_nodes:
    attribute_embeddings[i]=attribute_sketches[i]
    Q=bitarray(rando()<attribute_bitset_probability for j in range(attribute_dimension))
    R[i]=attribute_sketches[i]&Q
del(attribute_sketches)
got_R=dict()
for i in partitioned_nodes:
    for j in adjacency_list[i]:
        if j in partitioned_nodes:
            attribute_embeddings[i]=R[j]|attribute_embeddings[i]
        else:
            give_R[partition[j]][i]=R[i]
del(R)
del(adjacency_list)
tagger=1
for i in range(size):
    for j in range(size):
        if rank==i and i!=j:
            comm.send(give_R[j],dest=j,tag=tagger)
        elif rank==j and i!=j:
            got_R.update(comm.recv(source=i,tag=tagger))
        tagger+=1
del(give_R)
for i in to_get:
    attribute_embeddings[i[0]]=got_R[i[1]]|attribute_embeddings[i[0]]
del(got_R)
topology_embeddings=comm.gather(topology_embeddings)
attribute_embeddings=comm.gather(attribute_embeddings)
if rank==0:
    embeddings=[[] for i in range(nn)]
    for i in topology_embeddings:
        for k,l in i.items():
            embeddings[k]=l
    del(topology_embeddings)
    for i in attribute_embeddings:
        for k,l in i.items():
            embeddings[k]=embeddings[k]+l
    del(attribute_embeddings)
    time+=timeit.default_timer()-start_time
    print('Embedding Space = {0}B'.format(sys.getsizeof(embeddings)))
    print('Embedding Time = {0}s'.format(round(time,4)),end='')
    pickle.dump(embeddings,open('embeddingxx.pkl','wb'))
else:
    del(topology_embeddings)
    del(attribute_embeddings)

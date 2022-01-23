from mpi4py import MPI
import numpy as np
from scipy import sparse
from bitarray import util
import copy
from bitarray import bitarray
from random import random as rando
import pickle
import timeit
from itertools import islice,chain
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
    elif argument[0]=='l_t':
        l_t=int(argument[1][:-1])
    elif argument[0]=='l_a':
        l_a=int(argument[1][:-1])
    elif argument[0]=='f_t':
        f_t=int(argument[1][:-1])
    elif argument[0]=='f_a':
        f_a=int(argument[1][:-1])
    elif argument[0]=='fragments':
        f=int(argument[1][:-1])
    elif argument[0]=='nodes':
        nodes=int(argument[1][:-1])
    elif argument[0]=='edges':
        edges=int(argument[1][:-1])
    elif argument[0]=='attributes':
        attributes=int(argument[1][:-1])
del(argument,line)
file.close()
N_t=round((1-alpha)*N)
N_a=N-N_t
if rank==0:
    start_time=timeit.default_timer()
    Phi=np.random.randint(low=0,high=size,size=nodes)
else:
    Phi=np.empty(nodes,dtype='int')
comm.Bcast(Phi)
Phi_inverse=np.zeros(nodes,dtype=int)
j=0
for i in range(nodes):
    if Phi[i]==rank:
        Phi_inverse[j]=i
        j+=1
Phi_inverse=Phi_inverse[0:j]
if rank==0:
    time=timeit.default_timer()-start_time
edge_list=np.load('../../Datasets/'+graph+'/edge_list_preprocessed.npy')
in_edges=np.zeros((edges,2),dtype=int)
cross_edges=np.zeros((edges,2),dtype=int)
node_set=Phi_inverse.tolist()
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
if N_a>0:
    attribute_matrix=sparse.load_npz('../../Datasets/'+graph+'/attribute_matrix.npz')
    attribute_list={i:attribute_matrix[i].nonzero()[1] for i in Phi_inverse}
    del(attribute_matrix)
if rank==0:
    start_time=timeit.default_timer()
    if N_t>0:
        Pi_t=np.random.randint(low=0,high=N_t,size=nodes)
    if N_a>0:
        Pi_a=np.random.randint(low=0,high=N_a,size=attributes)
else:
    if N_t>0:
        Pi_t=np.empty(nodes,dtype=int)
    if N_a>0:
        Pi_a=np.empty(attributes,dtype=int)
if N_a>0:
    comm.Bcast(Pi_a)
    S_a={i:util.zeros(N_a) for i in Phi_inverse}
    for i in Phi_inverse:
        for j in attribute_list[i]:
            S_a[i][Pi_a[j]]=1
    del(Pi_a,attribute_list)
    if b_a==0:
        E_a=S_a
    else:
        for i in range(l_a):
            E_a=copy.deepcopy(S_a)
            if b_a==1:
                for j in in_edges:
                    E_a[j[1]]=S_a[j[0]]|E_a[j[1]]
                    E_a[j[0]]=S_a[j[1]]|E_a[j[0]]
                batch_S_a=[dict() for _ in range(size)]
                for j in cross_edges:
                    batch_S_a[Phi[j[1]]][j[0]]=S_a[j[0]]
                S_a_temp=dict()
                if f==1:
                    batch_S_a=comm.alltoall(batch_S_a)
                    for j in range(size):
                        S_a_temp.update(batch_S_a[j])
                else:
                    batch_sizes_S_a=[len(batch_S_a[j]) for j in range(size)]
                    batch_S_a=[iter(batch_S_a[j].items()) for j in range(size)]
                    for j in range(f):
                        mini_batch_S_a=[dict(islice(batch_S_a[k],batch_sizes_S_a[k]//(f-j))) for k in range(size)]
                        batch_sizes_S_a=[batch_sizes_S_a[k]-len(mini_batch_S_a[k]) for k in range(size)]
                        mini_batch_S_a=comm.alltoall(mini_batch_S_a)
                        for k in range(size):
                            S_a_temp.update(mini_batch_S_a[k])
                    del(batch_sizes_S_a,mini_batch_S_a)
                for j in cross_edges:
                    E_a[j[0]]=S_a_temp[j[1]]|E_a[j[0]]
                del(S_a_temp,batch_S_a)
            else:
                D_a=dict()
                for j in Phi_inverse:
                    Q_a=bitarray(rando()<b_a for _ in range(N_a))
                    D_a[j]=S_a[j]&Q_a
                for j in in_edges:
                    E_a[j[1]]=D_a[j[0]]|E_a[j[1]]
                    E_a[j[0]]=D_a[j[1]]|E_a[j[0]]
                batch_D_a=[dict() for _ in range(size)]
                for j in cross_edges:
                    batch_D_a[Phi[j[1]]][j[0]]=D_a[j[0]]
                D_a=dict()
                if f==1:
                    batch_D_a=comm.alltoall(batch_D_a)
                    for j in range(size):
                        D_a.update(batch_D_a[j])
                else:
                    batch_sizes_a=[len(batch_D_a[j]) for j in range(size)]
                    batch_D_a=[iter(batch_D_a[j].items()) for j in range(size)]
                    for j in range(f):
                        mini_batch_D_a=[dict(islice(batch_D_a[k],batch_sizes_a[k]//(f-j))) for k in range(size)]
                        batch_sizes_a=[batch_sizes_a[k]-len(mini_batch_D_a[k]) for k in range(size)]
                        mini_batch_D_a=comm.alltoall(mini_batch_D_a)
                        for k in range(size):
                            D_a.update(mini_batch_D_a[k])
                    del(batch_sizes_a,mini_batch_D_a)
                for j in cross_edges:
                    E_a[j[0]]=D_a[j[1]]|E_a[j[0]]
                del(D_a,Q_a,batch_D_a)
            S_a=E_a
            b_a/=f_a
    del(S_a)
if N_t>0:
    comm.Bcast(Pi_t)
    S_t={i:util.zeros(N_t) for i in Phi_inverse}
    for i in Phi_inverse:
        S_t[i][Pi_t[i]]=1
    for i in in_edges:
        S_t[i[0]][Pi_t[i[1]]]=1
        S_t[i[1]][Pi_t[i[0]]]=1
    for i in cross_edges:
        S_t[i[0]][Pi_t[i[1]]]=1
    del(Pi_t)
    if b_t==0:
        E_t=S_t
    else:
        for i in range(l_t):
            E_t=copy.deepcopy(S_t)
            if b_t==1:
                for j in in_edges:
                    E_t[j[1]]=S_t[j[0]]|E_t[j[1]]
                    E_t[j[0]]=S_t[j[1]]|E_t[j[0]]
                batch_S_t=[dict() for _ in range(size)]
                for j in cross_edges:
                    batch_S_t[Phi[j[1]]][j[0]]=S_t[j[0]]
                S_t_temp=dict()
                if f==1:
                    batch_S_t=comm.alltoall(batch_S_t)
                    for j in range(size):
                        S_t_temp.update(batch_S_t[j])
                else:
                    batch_sizes_S_t=[len(batch_S_t[j]) for j in range(size)]
                    batch_S_t=[iter(batch_S_t[j].items()) for j in range(size)]
                    for j in range(f):
                        mini_batch_S_t=[dict(islice(batch_S_t[k],batch_sizes_S_t[k]//(f-j))) for k in range(size)]
                        batch_sizes_S_t=[batch_sizes_S_t[k]-len(mini_batch_S_t[k]) for k in range(size)]
                        mini_batch_S_t=comm.alltoall(mini_batch_S_t)
                        for k in range(size):
                            S_t_temp.update(mini_batch_S_t[k])
                    del(batch_sizes_S_t,mini_batch_S_t)
                for j in cross_edges:
                    E_t[j[0]]=S_t_temp[j[1]]|E_t[j[0]]
                del(S_t_temp,batch_S_t)
            else:
                D_t=dict()
                for j in Phi_inverse:
                    Q_t=bitarray(rando()<b_t for _ in range(N_t))
                    D_t[j]=S_t[j]&Q_t
                for j in in_edges:
                    E_t[j[1]]=D_t[j[0]]|E_t[j[1]]
                    E_t[j[0]]=D_t[j[1]]|E_t[j[0]]
                batch_D_t=[dict() for _ in range(size)]
                for j in cross_edges:
                    batch_D_t[Phi[j[1]]][j[0]]=D_t[j[0]]
                D_t=dict()
                if f==1:
                    batch_D_t=comm.alltoall(batch_D_t)
                    for j in range(size):
                        D_t.update(batch_D_t[j])
                else:
                    batch_sizes_t=[len(batch_D_t[j]) for j in range(size)]
                    batch_D_t=[iter(batch_D_t[j].items()) for j in range(size)]
                    for j in range(f):
                        mini_batch_D_t=[dict(islice(batch_D_t[k],batch_sizes_t[k]//(f-j))) for k in range(size)]
                        batch_sizes_t=[batch_sizes_t[j]-len(mini_batch_D_t[j]) for j in range(size)]
                        mini_batch_D_t=comm.alltoall(mini_batch_D_t)
                        for k in range(size):
                            D_t.update(mini_batch_D_t[k])
                    del(batch_sizes_t,mini_batch_D_t)
                for j in cross_edges:
                    E_t[j[0]]=D_t[j[1]]|E_t[j[0]]
                del(D_t,Q_t,batch_D_t)
            S_t=E_t
            b_t/=f_t
    del(S_t)
del(in_edges,cross_edges,Phi_inverse)
if f==1:
    if N_t>0:
        E_t=comm.gather(E_t)
    if N_a>0:
        E_a=comm.gather(E_a)
else:
    if N_t>0:
        E_t_iterator=iter(E_t.items())
        buffer=len(E_t)
        E_t=[[] for _ in range(f)]
        for i in range(f):
            mini_batch_E_t=dict(islice(E_t_iterator,buffer//(f-i)))
            buffer-=len(mini_batch_E_t)
            E_t[i]=comm.gather(mini_batch_E_t)
        del(E_t_iterator,mini_batch_E_t)
    if N_a>0:
        E_a_iterator=iter(E_a.items())
        buffer=len(E_a)
        E_a=[[] for _ in range(f)]
        for i in range(f):
            mini_batch_E_a=dict(islice(E_a_iterator,buffer//(f-i)))
            buffer-=len(mini_batch_E_a)
            E_a[i]=comm.gather(mini_batch_E_a)
        del(E_a_iterator,mini_batch_E_a)
if rank==0:
    if f>1:
        if N_t>0:
            E_t=list(chain.from_iterable(E_t))
        if N_a>0:
            E_a=list(chain.from_iterable(E_a))
    emb=[[] for _ in range(nodes)]
    if N_t>0:
        for i in E_t:
            for k,l in i.items():
                emb[k]=l
        del(E_t)
    if N_a>0 and N_t==0:
        for i in E_a:
            for k,l in i.items():
                emb[k]=l
        del(E_a)
    elif N_a>0:
        for i in E_a:
            for k,l in i.items():
                emb[k]=emb[k]+l
        del(E_a)
    time+=timeit.default_timer()-start_time
    print('\nEmbedding Time = {0}s'.format(round(time,4)))
    pickle.dump(emb,open('../../Embeddings/'+graph+'_PBGENA_emb.pkl','wb'))
    del(emb)
else:
    if N_t>0:
        del(E_t)
    if N_a>0:
        del(E_a)
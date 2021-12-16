import pandas as pd
from scipy import sparse
import random
from bitarray import util
import timeit
from random import random as rando
from bitarray import bitarray
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
import math
from sklearn.multioutput import MultiOutputClassifier
import zipfile
import pickle
start_time=timeit.default_timer()
graph='citeseer'
classes='single'
N=2000
attribute_percentage=80
attribute_bitset_probability=0.9
topology_bitset_probability=0.4
attribute_level=3
topology_level=3
edge_list=pd.read_csv('../Datasets/'+graph+'/edgelist.txt',header=None,sep='\s+')
edge_list.drop_duplicates(inplace=True,ignore_index=True)
edge_list=edge_list.to_numpy()
e=set()
for i in range(edge_list.shape[0]):
    if (edge_list[i][0],edge_list[i][1]) not in e and (edge_list[i][1],edge_list[i][0]) not in e:
        e.add((edge_list[i][0],edge_list[i][1]))
e=list(e)
edge_list=np.array(e)
del(e)
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
    pickle.dump(labeled_nodes,open('labeled_node.pkl','wb'))
else:
    labels_read=pd.read_csv('../Datasets/'+graph+'/labels.txt',header=None,sep='\s+')
    if graph=='tweibo':
        labels=np.zeros(2320895,int)
    else:
        labels=np.zeros(labels_read.shape[0],int)
    for i in range(labels_read.shape[0]):
        labels[labels_read[0][i]]=labels_read[1][i]
del(labels_read)
attributes=sparse.load_npz('../Datasets/'+graph+'/attrs.npz')
nodes=len(labels)
print(graph,':')
print('#Nodes =',nodes)
print('#Edges =',edge_list.shape[0])
print('#Attributes =',attributes.shape[1])
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
topology_mapping=dict()
attribute_mapping=dict()
topology_dimension=int((100-attribute_percentage)/100*N)
attribute_dimension=N-topology_dimension
for i in range(nodes):
    topology_mapping[i]=random.randint(0,topology_dimension-1)
print('Topology Mapped')
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
for i in range(attributes.shape[1]):
    attribute_mapping[i]=random.randint(0,attribute_dimension-1)
print('Attributes Mapped')
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
topology_sketches=[]
for i in range(nodes):
    topology_sketches.append(util.zeros(topology_dimension))
for i in range(edge_list.shape[0]):
    topology_sketches[edge_list[i][0]][topology_mapping[edge_list[i][1]]]=1
    topology_sketches[edge_list[i][1]][topology_mapping[edge_list[i][0]]]=1
for i in range(nodes):
    topology_sketches[i][topology_mapping[i]]=1
print('Topology Sketched')
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
attribute_sketches=[]
for i in range(nodes):
    attribute_sketches.append(util.zeros(attribute_dimension))
n,a=attributes.nonzero()
del(attributes)
for i in range(len(n)):
    attribute_sketches[n[i]][attribute_mapping[a[i]]]=1
del(n)
del(a)
print('Attributes Sketched')
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
for kkk in range(topology_level):
    topology_embeddings=[]
    R=[]
    for i in range(nodes):
        topology_embeddings.append(topology_sketches[i])
        Q=bitarray(rando()<topology_bitset_probability for j in range(topology_dimension))
        R.append(topology_sketches[i]&Q)
    for i in range(edge_list.shape[0]):
        topology_embeddings[edge_list[i][1]]=R[edge_list[i][0]]|topology_embeddings[edge_list[i][1]]
    for i in range(edge_list.shape[0]):
        topology_embeddings[edge_list[i][0]]=R[edge_list[i][1]]|topology_embeddings[edge_list[i][0]]
    topology_sketches=topology_embeddings
    topology_bitset_probability/=2
del(topology_sketches)
print('Topology Propagated')
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
for kkk in range(attribute_level):
    attribute_embeddings=[]
    R=[]
    for i in range(nodes):
        attribute_embeddings.append(attribute_sketches[i])
        Q=bitarray(rando()<attribute_bitset_probability for j in range(attribute_dimension))
        R.append(attribute_sketches[i]&Q)
    for i in range(edge_list.shape[0]):
        attribute_embeddings[edge_list[i][1]]=R[edge_list[i][0]]|attribute_embeddings[edge_list[i][1]]
    for i in range(edge_list.shape[0]):
        attribute_embeddings[edge_list[i][0]]=R[edge_list[i][1]]|attribute_embeddings[edge_list[i][0]]
    attribute_sketches=attribute_embeddings
    attribute_bitset_probability/=2
del(attribute_sketches)
print('Attributes Propagated')
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
embeddings=[]
for i in range(nodes):
    embeddings.append(topology_embeddings[i]+attribute_embeddings[i])
print('Embedding Successful')
del(topology_embeddings)
del(attribute_embeddings)
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
start_time=timeit.default_timer()
del(R)
pickle.dump(embeddings,open('embedding.pkl','wb'))
pickle.dump(labels,open('label.pkl','wb'))
print('Saved')
del(topology_mapping)
del(attribute_mapping)
del(edge_list)
model=LogisticRegression(random_state=0,max_iter=1000)
if classes=='multi':
    model=OneVsRestClassifier(model)
    embeddings=[embeddings[i] for i in labeled_nodes]
    labels=[labels[i] for i in labeled_nodes]
if classes=='multi':
    l=MultiLabelBinarizer(sparse_output=True).fit_transform(labels)
    print('#Labels = ',l.shape[1])
    del(l)
else:
    print('#Labels = ',len(np.unique(labels)))
nodes=len(labels)
total=list(range(nodes))
test_select=random.sample(total,int(0.3*nodes))
train_select=list(set(total)-set(test_select))
X_train=sparse.lil_matrix((len(train_select),N),dtype=bool)
Y_train=[]
i=0
for j in train_select:
    for k in range(N):
        if embeddings[j][k]==1:
            X_train[i,k]=1
    i+=1
    Y_train.append(labels[j])
X_train=X_train.tocsr()
X_test=sparse.lil_matrix((len(test_select),N),dtype=bool)
Y_test=[]
i=0
for j in test_select:
    for k in range(N):
        if embeddings[j][k]==1:
            X_test[i,k]=1
    i+=1
    Y_test.append(labels[j])
X_test=X_test.tocsr()
del(embeddings)
if classes=='multi':
    binarizer=MultiLabelBinarizer(sparse_output=True)
    Y_train=binarizer.fit_transform(Y_train)
    Y_test=binarizer.transform(Y_test)
model.fit(X_train,Y_train)
predicted=model.predict(X_test)
macro=f1_score(Y_test,predicted,average='macro')
micro=f1_score(Y_test,predicted,average='micro')
print('Macro F1 =',macro)
print('Micro F1 =',micro)
elapsed=timeit.default_timer()-start_time
print('Time taken = {0}s'.format(round(elapsed,4)))
import warnings
import argparse
import os
import numpy as np
import pickle
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import timeit
class NodeClassifier(object):
    def __init__(self,graph,algorithm,multi,tr):
        print('\nSetting up Node Classifier...')
        assert os.path.isdir('../../Datasets/'+graph),'Folder for {0} network does not exist'.format(graph)
        self.__graph=graph
        assert isinstance(multi,bool),'multi should be boolean'
        self.__multi=multi
        if self.__multi==True:
            assert os.path.isfile('../../Datasets/'+self.__graph+'/label_array.npz'),'Label array file does not exist for {0} network'.format(self.__graph)
            self.__label_array=sparse.load_npz('../../Datasets/'+self.__graph+'/label_array.npz')
        else:
            assert os.path.isfile('../../Datasets/'+self.__graph+'/label_array.npy'),'Label array file does not exist for {0} network'.format(self.__graph)
            self.__label_array=np.load('../../Datasets/'+self.__graph+'/label_array.npy')
        assert os.path.isfile('../../Embeddings/'+self.__graph+'_'+algorithm+'_emb.pkl'),'Either network name {0} or algorithm name {1} is wrong'.format(self.__graph,algorithm)
        self.__algorithm=algorithm
        self.__emb=pickle.load(open('../../Embeddings/'+self.__graph+'_'+self.__algorithm+'_emb.pkl','rb'))
        if self.__algorithm=='BGENA' or self.__algorithm=='PBGENA':
            for i in range(len(self.__emb)):
                self.__emb[i]=np.frombuffer(self.__emb[i].unpack(),dtype=bool)
            self.__emb=np.array(self.__emb)
        assert tr>0 and tr<1,'Training Ratio must be in the range (0,1)'
        self.__tr=tr
    def remove_unlabeled_instances(self):
        print('\nRemoving unlabeled instances...')
        print('\n{0}:'.format(self.__graph))
        if self.__multi==True:
            unlabeled_nodes=[]
            labels=[]
            for i in range(self.__label_array.shape[0]):
                if self.__label_array[i].nonzero()[1].size==0:
                    unlabeled_nodes.append(i)
                else:
                    labels.append(self.__label_array[i].nonzero()[1])
            self.__emb=np.delete(self.__emb,unlabeled_nodes,axis=0)
            self.__label_array=MultiLabelBinarizer(sparse_output=True).fit_transform(labels)
            print('#Labels =',self.__label_array.shape[1])
        else:
            print('#Labels =',len(np.unique(self.__label_array)))
    def perform(self):
        print('\nTraining Classifier...')
        self.__X_train,self.__X_test,self.__Y_train,self.__Y_test=train_test_split(self.__emb,self.__label_array,test_size=1-self.__tr)
        self.__model=LogisticRegression(random_state=0,max_iter=1000)
        if self.__multi==True:
            self.__model=OneVsRestClassifier(self.__model)
        self.__model.fit(self.__X_train,self.__Y_train)
        self.__predicted=self.__model.predict(self.__X_test)
        self.__macro=f1_score(self.__Y_test,self.__predicted,average='macro')
        self.__micro=f1_score(self.__Y_test,self.__predicted,average='micro')
        print('\nPerformance:')
        print('Macro F1 =',self.__macro)
        print('Micro F1 =',self.__micro)
if __name__=='__main__':
    warnings.filterwarnings('ignore')
    parser=argparse.ArgumentParser(description='Node Classification')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--algorithm',type=str,default='PBGENA',help='Algorithm Name')
    parser.add_argument('--multi',type=bool,default=False,help='Multi-labelled Network?')
    parser.add_argument('--tr',type=float,default=0.7,help='Training Ratio')
    args=parser.parse_args()
    node_classifier=NodeClassifier(graph=args.graph,algorithm=args.algorithm,multi=args.multi,tr=args.tr)
    node_classifier.remove_unlabeled_instances()
    start_time=timeit.default_timer()
    node_classifier.perform()
    elapsed=timeit.default_timer()-start_time
    print('\nClassification Time = {0}s\n'.format(round(elapsed,4)))
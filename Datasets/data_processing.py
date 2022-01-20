import argparse
import os
import tarfile
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
class DataProcessor(object):
    def __init__(self,file,graph,multi):
        print('\nSetting up Data Processor...')
        assert os.path.isfile(file+'.attr.tar.gz'),'File {0}.attr.tar.gz of compressed network does not exist'.format(file)
        self.__file=file
        assert isinstance(graph,str),'Graph Folder Name must be a string'
        self.__graph=graph
        assert isinstance(multi,bool),'multi should be boolean'
        self.__multi=multi
    def extract(self):
        print('\nExtracting Files...')
        tarfile.open(self.__file+'.attr.tar.gz').extractall(self.__graph)
    def create_edge_list(self):
        print('\nCreating Edge List...')
        self.__edge_list=pd.read_csv(self.__graph+'/edgelist.txt',header=None,sep='\s+')
        self.__edge_list=self.__edge_list.to_numpy()
        np.save(self.__graph+'/edge_list',self.__edge_list)
    def create_attribute_matrix(self):
        print('\nCreating Attribute Matrix...')
        self.__attribute_matrix=pickle.load(open(self.__graph+'/attrs.pkl','rb'),encoding='latin1')
        sparse.save_npz(self.__graph+'/attribute_matrix',self.__attribute_matrix)
    def create_label_array(self):
        print('\nCreating Label Array...')
        if self.__multi==True:
            labels_read=open(self.__graph+'/labels.txt','r+').readlines()
            self.__label_array=[set() for _ in range(len(labels_read))]
            for i in labels_read:
                j=i.strip().split()
                j=[int(k) for k in j]
                self.__label_array[j[0]]=set(j[1:])
            self.__label_array=MultiLabelBinarizer(sparse_output=True).fit_transform(self.__label_array)
            sparse.save_npz(self.__graph+'/label_array',self.__label_array)
        else:
            labels_read=pd.read_csv(self.__graph+'/labels.txt',header=None,sep='\s+')
            labels_read=labels_read.to_numpy()
            self.__label_array=np.zeros(self.__attribute_matrix.shape[0],int)
            for i in labels_read:
                self.__label_array[i[0]]=i[1]
            np.save(self.__graph+'/label_array',self.__label_array)
    def delete_temporary_files(self):
        print('\nDeleting Temporary Files...\n')
        os.remove(self.__graph+'/edgelist.txt')
        os.remove(self.__graph+'/attrs.pkl')
        os.remove(self.__graph+'/labels.txt')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--file',type=str,help='File Name')
    parser.add_argument('--graph',type=str,help='Network Name')
    parser.add_argument('--multi',type=bool,default=False,help='Multi-labelled Network?')
    args=parser.parse_args()
    data_processor=DataProcessor(file=args.file,graph=args.graph,multi=args.multi)
    data_processor.extract()
    data_processor.create_edge_list()
    data_processor.create_attribute_matrix()
    data_processor.create_label_array()
    data_processor.delete_temporary_files()
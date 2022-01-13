# PBGENA
## Parallelized Binary embedding GENerator for Attributed graphs

![PBGENA Snapshot](https://github.com/tapadeep/PBGENA/blob/main/Examples/PBGENA_snapshot.png)

### Get PBGENA:
```
git clone https://github.com/tapadeep/PBGENA.git
```

### Satisfy all requirements for PBGENA:
```
pip install -r PBGENA/requirements.txt
```

### Execute BGENA:

#### Perform Node Embedding:
```
cd PBGENA/Code/Algorithm
python BGENA.py --graph Facebook --N 2000 --alpha 0.5 --b_a 0.7 --b_t 0.6
```

#### Perform Node Classification:
```
cd 'PBGENA/Code/Node Classification'
python node_classification.py --graph Facebook --algorithm BGENA --tr 0.7 --multi True
```
Ignore the ```--multi``` flag for graphs that are not multi-labeled.


#### Perform Link Prediction:
```
cd 'PBGENA/Code/Link Prediction'
python link_prediction.py --graph Facebook --algorithm BGENA --erf 0.3 --N 2000 --alpha 0.8 --b_a 1 --b_t 0
```

### Execute PBGENA:

#### Perform Node Embedding:
```
cd PBGENA/Code/Algorithm
python PBGENA.py --graph PubMed --N 8000 --alpha 0.65 --b_a 0 --b_t 0.8 --p 6
```

#### Perform Node Classification:
```
cd 'PBGENA/Code/Node Classification'
python node_classification.py --graph PubMed --algorithm PBGENA --tr 0.7
```

### Relevant Hyperparameters:

#### Node Classification Hyperparameters:
Graph | α | b_a | b_t |
:---: | :---: | :---: | :---: |
Wikipedia | 0.85 | 0.00 | 0.20 |
Cora | 0.60 | 0.80 | 0.80 |
CiteSeer | 0.80 | 0.90 | 0.40 |
Facebook | 0.50 | 0.70 | 0.60 |
BlogCatalog | 0.60 | 0.00 | 0.00 |
Flickr | 0.90 | 0.00 | 0.85 |
PubMed | 0.65 | 0.00 | 0.80 |
PPI | 0.10 | 0.95 | 0.50 |
Twitter | 0.60 | 0.00 | 0.00 |
Google+ | 0.15 | 0.50 | 0.00 |
Reddit | 0.10 | 0.00 | 0.00 |
TWeibo | 0.60 | 0.00 | 0.00 |
MAKG | 0.85 | 0.86 | 0.60 |

#### Link Prediction Hyperparameters:
Graph | α | b_a | b_t |
:---: | :---: | :---: | :---: |
Wikipedia | 0.95 | 0.20 | 0.20 |
Cora | 0.60 | 0.30 | 0.40 |
CiteSeer | 0.90 | 0.20 | 0.40 |
Facebook | 0.80 | 1.00 | 0.00 |
BlogCatalog | 0.90 | 0.20 | 0.00 |
Flickr | 0.90 | 0.00 | 0.00 |
PubMed | 0.95 | 0.20 | 0.20 |
PPI | 0.10 | 0.00 | 0.10 |
Twitter | 0.95 | 0.00 | 0.20 |
Google+ | 0.90 | 0.05 | 0.10 |
Reddit | 0.95 | 0.20 | 0.00 |
TWeibo | 0.95 | 0.00 | 0.00 |
MAKG | 0.85 | 0.86 | 0.60 |

### Create your own network:
To create your own network, you need three files: ```edge_list.npy```, ```attribute_matrix.npz```, and ```label_array.npy``` or ```label_array.npz``` (depending on whether the graph is single-labeled or multi-labeled). Make sure all your vertex IDs are in the range _(0, nodes-1)_. Create a numpy array of the edges of shape _(edges, 2)_ and save that file as ```edge_list.npy```. Store the attributes as a sparse CSR-matrix of shape _(nodes, attributes)_. To run PBGENA without attributes, simply create an empty attribute matrix for the non-attributed graph of proper shape, and set _α=0_. Save the attribute file as ```attribute_matrix.npz```. The file for label array is required only for node classification, and can be ignored to perform link prediction on unlabeled graphs. If the graph is single-labeled, the file ```label_array.npy``` is a simple 1D array where the _i_'th number denotes the label for node _i_. For multi-labeled graphs, ```label_array.npz``` is a [MultiLabelBinarized](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) CSR-matrix of shape _(nodes, labels)_.

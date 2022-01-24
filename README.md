# PBGENA
## Parallelized Binary embedding GENerator for Attributed graphs
A Sketch-based Approach towards Scalable and Efficient Attributed Network Embedding

<img src="https://github.com/tapadeep/PBGENA/blob/main/Examples/PBGENA_snapshot.png" width="575">

### Key Features:
1. BGENA is a super-fast sketch-based ANE solver, which uses [BinSketch](https://doi.org/10.1109/ICDM.2019.00061) coupled with a novel edge propagation mechanism.
2. PBGENA, Parallelized BGENA, works with MPI to leverage a system's multi-core architecture to accelerate the embedding process further.
3. PBGENA's binary embeddings allow efficient [bitarray](https://github.com/ilanschnell/bitarray)/[sparse-matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html) storage.
4. PBGENA embeddings achieve state-of-the-art performance in graph analysis tasks like node classification and link prediction.
5. PBGENA is highly flexible and can work with just the topology or the attributes of the graph and enables turning off propagation altogether.

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
python -B link_prediction.py --graph Facebook --algorithm BGENA --erf 0.3 --N 2000 --alpha 0.8 --b_a 1 --b_t 0
```

### Execute PBGENA:

#### Perform Node Embedding:
```
cd PBGENA/Code/Algorithm
python PBGENA.py --graph PubMed --N 8000 --alpha 0.65 --b_a 0 --b_t 0.8 --p 6
```
The system must have Open MPI, MPICH, or Microsoft MPI installed.

#### Perform Node Classification:
```
cd 'PBGENA/Code/Node Classification'
python node_classification.py --graph PubMed --algorithm PBGENA --tr 0.7
```

#### Perform Link Prediction:
```
cd 'PBGENA/Code/Link Prediction'
python -B link_prediction.py --graph PubMed --algorithm PBGENA --erf 0.3 --N 8000 --alpha 0.95 --b_a 0.2 --b_t 0.2 --p 6
```

### Execute Baselines:

#### Perform Node Embedding:
```
cd PBGENA/Code/Algorithm
python Baseline.py --graph CiteSeer --algorithm FeatherNode --reduction_dimensions 25 --eval_points 5 --order 1 --theta_max 1
```

#### Perform Node Classification:
```
cd 'PBGENA/Code/Node Classification'
python node_classification.py --graph CiteSeer --algorithm FeatherNode --tr 0.7
```

#### Perform Link Prediction:
```
cd 'PBGENA/Code/Link Prediction'
python -B link_prediction.py --graph CiteSeer --algorithm FeatherNode --erf 0.3 --reduction_dimensions 25 --eval_points 5 --order 1 --theta_max 1
```

### Relevant Flags:
Flag | Type | Description | Range | Default |
:---: | :---: | :--- | :---: | :---: |
```--algorithm``` | ```STRING``` | Embedding Method to be used | _{BGENA, PBGENA,<br>Baselines}_ | _PBGENA_ |
```--alpha``` | ```FLOAT``` | Fraction of the Embedding Dimension<br>to be used for attributes | _\[0, 1\]_ | _-_ |
```--b_a``` | ```FLOAT``` | Attribute Bitset Probability | _\[0, 1\]_ | _-_ |
```--b_t``` | ```FLOAT``` | Topology Bitset Probability | _\[0, 1\]_ | _-_ |
```--erf``` | ```FLOAT``` | Fraction of edges to be removed<br>for link prediction | _(0, 1)_ | _0.3_ |
```--f``` | ```INTEGER``` | Fragment parameter for PBGENA<br>communication calls | _ℕ_ | _1_ |
```--f_a``` | ```FLOAT``` | Reduction fraction for b_a each pass | _\[1, ∞)_ | _2_ |
```--f_t``` | ```FLOAT``` | Reduction fraction for b_t each pass | _\[1, ∞)_ | _2_ |
```--graph``` | ```STRING``` | Network to be used for embedding | _Any Network_ | _-_ |
```--l_a``` | ```INTEGER``` | Number of passes of attribute<br>edge propagation | _ℕ_ | _1_ |
```--l_t``` | ```INTEGER``` | Number of passes of topology<br>edge propagation | _ℕ_ | _1_ |
```--multi``` | ```BOOLEAN``` | Identify if the graph is Multi-Labeled | _{True, False}_ | _False_ |
```--N``` | ```INTEGER``` | Embedding Dimension | _ℕ_ | _2000_ |
```--p``` | ```INTEGER``` | Number of workers to use | _ℕ_ | _32_ |
```--tr``` | ```FLOAT``` | Training Ratio for Node Classification | _(0, 1)_ | _0.7_ |

Some of the others are standard flags, and the remaining flag descriptions are available with the python graph learning framework [Karate Club](https://karateclub.readthedocs.io/).

### Examples:
An online [Google Colaboratory Notebook](https://colab.research.google.com/drive/1BxVSlK0UNK4e1-5S6Ntw0HhbiAqSv9P5?usp=sharing) demonstrates PBGENA's working, and other examples are in the ```Examples``` folder.

### Relevant Hyperparameters:

#### Node Classification Hyperparameters:
Graph | α | b_a | b_t |
:--- | :---: | :---: | :---: |
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
:--- | :---: | :---: | :---: |
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

### Datasets:
Graph | #Vertices | #Edges | #Attributes | #Labels | Multi-<br>labeled? |
:--- | :--- | :--- | :--- | :--- | :---: |
Wikipedia | 2,405 | 11,596 | 4,973 | 17 | No |
Cora | 2,708 | 5,278 | 1,433 | 7 | No |
CiteSeer | 3,312 | 4,536 | 3,703 | 6 | No |
Facebook | 4,039 | 88,234 | 1,283 | 193 | Yes |
BlogCatalog | 5,196 | 171,743 | 8,189 | 6 | No |
Flickr | 7,575 | 239,738 | 12,047 | 9 | No |
PubMed | 19,717 | 44,324 | 500 | 3 | No |
PPI | 56,944 | 793,632 | 50 | 121 | Yes |
Twitter | 81,306 | 1,342,296 | 216,839 | 4,065 | Yes |
Google+ | 107,614 | 12,238,285 | 15,907 | 468 | Yes |
Reddit | 232,965 | 57,307,946 | 602 | 41 | No |
TWeibo | 2,320,895 | 50,133,369 | 1,657 | 9 | No |
MAKG | 59,249,719 | 976,901,586 | 7,211 | 100 | Yes |

Some preprocessed networks are added to the ```Datasets``` folder, and the remaining can be downloaded [online](https://drive.google.com/drive/folders/16qCQhylABkaLD-RlBgRQlDUgu6a2HaZS?usp=sharing). All of the datasets are obtained from [Dr. Jieming Shi's website](https://www4.comp.polyu.edu.hk/~jiemshi/datasets.html). To generate files suitable for PBGENA with data from their website, use:
```
cd PBGENA/Datasets
python data_processing.py --file cora --graph Cora
```
This converts ```cora.attr.tar.gz``` to the ```Cora``` folder containing relevant files which can be used to train PBGENA. Use the ```--multi True``` flag for converting multi-labeled networks.

### Bechmarked Performance

<table>
  <tr>
    <td colspan='2'>Node Classification (%)</td>
    <td colspan='2'>Link Prediction (%)</td>
    <td>Time (s)</td>
  </tr>
  <tr>
    <td>Macro-F1</td>
    <td>Micro-F1</td>
    <td>AUC-ROC</td>
    <td>Average Precision</td>
    <td></td>
  </tr>
</table>

### Create your own network:
To create your network, you need three files: ```edge_list.npy```, ```attribute_matrix.npz```, and ```label_array.npy``` or ```label_array.npz``` (depending on whether the graph is single-labeled or multi-labeled). Ensure all your vertex IDs are in the range _(0, nodes-1)_. Create a numpy array of the edges of shape _(edges, 2)_ and save that file as ```edge_list.npy```. Store the attributes as a sparse CSR matrix of shape _(nodes, attributes)_. To run PBGENA without attributes, create an empty attribute matrix for the non-attributed graph of the proper shape and set _α=0_. Save the attribute file as ```attribute_matrix.npz```. The file for label array is required only for node classification and can be ignored to perform link prediction on unlabeled graphs. If the graph is single-labeled, the file ```label_array.npy``` is a simple 1D array where the _i_'th number denotes the label for node _i_. For multi-labeled graphs, ```label_array.npz``` is a [MultiLabelBinarized](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) CSR-matrix of shape _(nodes, labels)_. Finally, put these three files in a folder named with the graph name and add it to the ```Datasets``` folder.

# PBGENA
## Parallelized Binary embedding GENerator for Attributed graphs

### Get PBGENA:
```
git clone https://github.com/tapadeep/PBGENA.git
```

### Satisfy all requirements for PBGENA:
```
pip install -r PBGENA/requirements.txt
```

### Perform Node Embedding:
```
cd PBGENA/Code/Algorithm
python BGENA.py --graph Facebook --N 2000 --alpha 0.5 --b_a 0.7 --b_t 0.6
```

### Perform Node Classification:
```
cd 'PBGENA/Code/Node Classification'
python node_classification.py --graph Facebook --algorithm BGENA --tr 0.7 --multi True
```
Ignore the ```--multi``` flag for graphs that are not multi-labeled.


### Perform Link Prediction:
```
cd 'PBGENA/Code/Link Prediction'
python link_prediction.py --graph Facebook --algorithm BGENA --erf 0.3 --N 2000 --alpha 0.8 --b_a 1 --b_t 0
```

### Relevant Hyperparameters
Graph | $`\alpha`$ | $`b_a`$ | $`b_t`$ |
--- | --- | --- | --- |
Wikipedia | 0.85 | 0.00 | 0.20 |

### Create your own network:
To create your own network, you need three files: ```attribute_matrix.npz```, ```edge_list.npy```, and ```label_array.npy``` or ```label_array.npz``` (depending on whether the graph is single labeled or multi-labeled). 

Task II: Classical Graph Neural Network (GNN) 

OBJECTIVE: To use graph neural networks (GNNs) for Quarks/Gluon jet classification 

DATASET: ParticleNet point cloud data (https://zenodo.org/records/3164691/files/QG_jets_1.npz?download=1)

The two graph based of my choice are:
1.) Graph Convolutional Network (GCN)
2.) Graph Attention Network (GAT)

Graph Convolutional Network is effective (GCN) for quark/gluon jet classification because to it aggregates data from neighboring nodes (i.e particles) equally to learn local jet structure.

Graph Attention Network (GAT) are widely used for classification uses attention to aggregate data from neighbors differently and highlights peculiar nodal interaction in the jet structure.

OBSERVATION: 
According to training and evaluation, GCN has an accuracy of 0.665 and ROC-AUC of 0.704; and GAT has an accuracy of  0.733 and ROC-AUC of 0.814.
while they both learn from aggregating neighbor information, GCN is simpler and and faster while GAT focuses on key particles and is more accurate.

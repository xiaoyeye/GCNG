# GCNG
using graph convolutional neural network and spaital transcriptomics data to infer cell-cell interactions
# Title, GCNG: Graph convolutional networks for inferring cell-cell interactions
## date: Nov 1, 2019

># 1, GCNG
![](https://github.com/xiaoyeye/GCNG/blob/master/GCNG.bmp)

GCNG for extracellular gene relationship inference. (A) GCNG model using spatial single cell expression data. A binary cell adjacent matrix and an expression matrix are extracted from spatial data. After normalization, both matrices are fed into the graph convolutional network. (B) Training and test data separation and generation strategy. The known ligand and receptor genes can form complicated directed networks. For cross validation, all ligand and receptors are separated exclusively as training and test gene sets, and only gene pairs where both genes are in training (test) are used for training (test). To balance the dataset, each positive ligand-receptor (La; Rb) gene pair with label 1 will have a negative pair sample (La; Rx) with label 0 where Rx was randomly selected from all training (test) receptor genes which are not interacting with La in training (test).

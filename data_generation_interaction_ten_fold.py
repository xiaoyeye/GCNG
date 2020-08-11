import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
####################  get the whole training dataset

current_path = os.path.abspath('.')
cortex_svz_cellcentroids = pd.read_csv(current_path+'/seqfish_plus/cortex_svz_cellcentroids.csv')
############# get batch adjacent matrix
cell_view_list = []
for view_num in range(7):
    cell_view = cortex_svz_cellcentroids[cortex_svz_cellcentroids['Field of View']==view_num]
    cell_view_list.append(cell_view)

############ the distribution of distance
distance_list_list = []
distance_list_list_2 = []
print ('calculating distance matrix, it takes a while')
for view_num in range(7):
    print (view_num)
    cell_view = cell_view_list[view_num]
    distance_list = []
    for j in range(cell_view.shape[0]):
        for i in range (cell_view.shape[0]):
            if i!=j:
                distance_list.append(np.linalg.norm(cell_view.iloc[j][['X','Y']]-cell_view.iloc[i][['X','Y']]))
    distance_list_list = distance_list_list + distance_list
    distance_list_list_2.append(distance_list)

# np.save(current_path+'/seqfish_plus/distance_array.npy',np.array(distance_list_list))
###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
from scipy import sparse
import pickle
import spektral
import scipy.linalg
distance_array = np.array(distance_list_list)
for threshold in [100]:#[100,140,180,210,220,260]:#range (210,211):#(100,400,40):
    num_big = np.where(distance_array<threshold)[0].shape[0]
    print (threshold,num_big,str(num_big/(913*2)))
    distance_matrix_threshold_I_list = []
    distance_matrix_threshold_W_list = []
    from sklearn.metrics.pairwise import euclidean_distances
    for view_num in range (7):
        cell_view = cell_view_list[view_num]
        distance_matrix = euclidean_distances(cell_view[['X','Y']], cell_view[['X','Y']])
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = distance_matrix[i,j]
        distance_matrix_threshold_I_list.append(distance_matrix_threshold_I)
        distance_matrix_threshold_W_list.append(distance_matrix_threshold_W)
    whole_distance_matrix_threshold_I = scipy.linalg.block_diag(distance_matrix_threshold_I_list[0],
                                                                distance_matrix_threshold_I_list[1],
                                                                distance_matrix_threshold_I_list[2],
                                                                distance_matrix_threshold_I_list[3],
                                                                distance_matrix_threshold_I_list[4],
                                                                distance_matrix_threshold_I_list[5],
                                                                distance_matrix_threshold_I_list[6])
    ############### get normalized sparse adjacent matrix
    distance_matrix_threshold_I_N = spektral.utils.normalized_adjacency(whole_distance_matrix_threshold_I, symmetric=True)
    # distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I) ## do not normalize adjcent matrix
    distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I)
    distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
    with open(current_path+'/seqfish_plus/whole_FOV_distance_I_N_crs_'+str(threshold), 'wb') as fp:
        pickle.dump(distance_matrix_threshold_I_N_crs, fp)
    distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I) ## do not normalize adjcent matrix
    distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
    with open(current_path+'/seqfish_plus/whole_FOV_distance_I_N_crs_'+str(threshold), 'wb') as fp:
        pickle.dump(distance_matrix_threshold_I_crs, fp)
#
# generate graph matrix where each cell only connect itself
# import spektral
# from scipy import sparse
# import pickle
# import numpy as np
# whole_distance_matrix_threshold_I_none = np.zeros((913,913))
# whole_distance_matrix_threshold_I_none_N = spektral.utils.normalized_adjacency(whole_distance_matrix_threshold_I_none, symmetric=True)
# whole_distance_matrix_threshold_I_none_N = np.float32(whole_distance_matrix_threshold_I_none_N)
# whole_distance_matrix_threshold_I_none_N_crs = sparse.csr_matrix(whole_distance_matrix_threshold_I_none_N)
# with open(current_path+'/seqfish_plus/whole_FOV_distance_I_none_N_crs', 'wb') as fp:
#     pickle.dump(whole_distance_matrix_threshold_I_none_N_crs, fp)

########### read ligand receptor database
ligand_list = pd.read_csv(current_path+'/LR_database/ligand_list2.txt',header  = None)
receptor_list = pd.read_csv(current_path+'/LR_database/receptor_list2.txt',header  = None)
LR_pairs = pd.read_csv(current_path+'/LR_database/ligand_receptor_pairs2.txt',header  = None,sep ='\t')

#####################
cortex_svz_counts = pd.read_csv(current_path+'/seqfish_plus/cortex_svz_counts.csv')
cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)] ## gene expression normalization
cortex_svz_cellcentroids = pd.read_csv(current_path+'/seqfish_plus/cortex_svz_cellcentroids.csv')
# cortex_svz_counts_N_FOV = cortex_svz_counts_N
#######################

gene_list =[i.lower() for i in list(cortex_svz_counts)]

non_LR_list = [i for i in gene_list if i not in list(ligand_list.iloc[:,0]) and i not in list(receptor_list.iloc[:,0])]
ovlp_ligand_list = [i for i in gene_list if i in list(ligand_list.iloc[:,0])]
ovlp_receptor_list = [i for i in gene_list if i in list(receptor_list.iloc[:,0])]

count = 0
h_LR = defaultdict(list)
for LR_pair_index in range(LR_pairs.shape[0]):
    ligand, receptor =  LR_pairs.iloc[LR_pair_index]
    if ligand in gene_list and receptor in gene_list:
        h_LR[ligand].append(receptor)
        count = count + 1

################### generate training dataset containing both postive and negative samples, where negative samples still in the ligand and receptor set, but not in pair set
############################################# to split the LR database completely

def generate_LR_pairs (h_LR_original,sub_ligand_list, sub_receptor_list,cortex_svz_counts_N):
    h_LR = defaultdict(list)
    for ligand in h_LR_original.keys():
        if ligand in sub_ligand_list:
            for receptor in h_LR_original[ligand]:
                if receptor in sub_receptor_list:
                    h_LR[ligand].append(receptor)
    import random
    random.seed(0)
    count = 0
    gene_pair_list  = []
    X_data = []
    Y_data = []
    sub_ligand_list_ovlp = list(h_LR.keys())
    for ligand in sub_ligand_list_ovlp:
        for receptor in h_LR[ligand]:
            gene_pair_list.append(ligand + '\t' + receptor)
            cell_LR_expression = np.array(cortex_svz_counts_N[[ligand, receptor]]) # postive sample
            X_data.append(cell_LR_expression)
            Y_data.append(1)
            ############## get negative samples
            non_pair_receptor_list = [i for i in sub_receptor_list if i not in h_LR[ligand]]
            random.seed(count)
            random_receptor = random.sample(non_pair_receptor_list, 1)[0]
            gene_pair_list.append(ligand + '\t' + random_receptor)
            cell_LR_expression = np.array(cortex_svz_counts_N[[ligand, random_receptor]])
            X_data.append(cell_LR_expression)
            Y_data.append(0)
            count = count + 1
    ligand_record = sub_ligand_list_ovlp[0]
    gene_pair_index = [0]
    count = 0
    for gene_pair in gene_pair_list:
        ligand = gene_pair.split('\t')[0]
        if ligand == ligand_record:
            count = count + 1
        else:
            gene_pair_index.append(count)
            ligand_record = ligand
            count = count + 1
    gene_pair_index.append(count)
    X_data_array = np.array(X_data)
    Y_data_array = np.array(Y_data)
    gene_pair_list_array = np.array(gene_pair_list)
    gene_pair_index_array = np.array(gene_pair_index)
    return (X_data_array,Y_data_array,gene_pair_list_array,gene_pair_index_array) ## x data, y data, gene pair name, index to separate pairs by ligand genes


########## ten fold cross validation data generation
ovlp_ligand_list_cons = ovlp_ligand_list
ovlp_receptor_list_cons = ovlp_receptor_list
import random
random.seed(1)
ovlp_ligand_list = random.sample(ovlp_ligand_list_cons,len(ovlp_ligand_list))
random.seed(1)
ovlp_receptor_list = random.sample(ovlp_receptor_list_cons,len(ovlp_receptor_list))
for test_indel in range(1,11): ################## ten fold cross validation
    print (test_indel)
    ######### completely separate ligand and recpetor genes as mutually  exclusive train and test set
    whole_ligand_index = [i for i in range(len(ovlp_ligand_list))]
    test_ligand = [i for i in range (int(np.ceil((test_indel-1)*0.1*len(ovlp_ligand_list))),int(np.ceil(test_indel*0.1*len(ovlp_ligand_list))))]
    train_ligand= [i for i in whole_ligand_index if i not in test_ligand]
    whole_receptor_index = [i for i in range(len(ovlp_receptor_list))]
    test_receptor = [i for i in range(int(np.ceil((test_indel - 1) * 0.1 * len(ovlp_receptor_list))),int(np.ceil(test_indel * 0.1 * len(ovlp_receptor_list))))]
    train_receptor = [i for i in whole_receptor_index if i not in test_receptor]
    X_data_array_train, Y_data_array_train, gene_pair_list_array_train, gene_pair_index_array_train = generate_LR_pairs (h_LR,np.array(ovlp_ligand_list)[train_ligand], np.array(ovlp_receptor_list)[train_receptor],cortex_svz_counts_N)
    X_data_array_test, Y_data_array_test, gene_pair_list_array_test, gene_pair_index_array_test = generate_LR_pairs(h_LR, np.array(ovlp_ligand_list)[test_ligand], np.array(ovlp_receptor_list)[test_receptor], cortex_svz_counts_N)
    if not os.path.isdir(current_path + '/rand_1_10fold/'):
        os.makedirs(current_path + '/rand_1_10fold/')
    np.save(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_X_data_array.npy', X_data_array_train)
    np.save(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_Y_data_array.npy', Y_data_array_train)
    np.save(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_gene_pair_list_array.npy', gene_pair_list_array_train)
    np.save(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_gene_pair_index_array.npy', gene_pair_index_array_train)
    np.save(current_path+'/rand_1_10fold/' + str(test_indel) + '_test_X_data_array.npy',X_data_array_test)
    np.save(current_path+'/rand_1_10fold/' + str(test_indel) + '_test_Y_data_array.npy',Y_data_array_test)
    np.save(current_path+'/rand_1_10fold/' + str(test_indel) + '_test_gene_pair_list_array.npy',gene_pair_list_array_test)
    np.save(current_path+'/rand_1_10fold/' + str(test_indel) + '_test_gene_pair_index_array.npy',gene_pair_index_array_test)


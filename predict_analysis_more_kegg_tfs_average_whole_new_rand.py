from sklearn.metrics import precision_recall_curve
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
data_augmentation = False
# num_predictions = 20
batch_size = 256
num_classes = 3
epochs = 200
data_augmentation = False
# num_predictions = 20
model_name = 'keras_cnn_trained_model_shallow.h5'
# The data, shuffled and split between train and test sets:
current_path = os.path.abspath('.')


save_dir = os.path.join(os.getcwd(),'_Ycv_LR_as_nega_rg_5-7_lr_1-6_e100')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    # plt.grid()

mean_fpr = np.linspace(0, 1, 100)

y_testy = np.empty([0,1])
y_predicty = np.empty([0,1])
#count_setx = pd.read_table('/home/yey3/nn_project2/data/human_brain/pathways/kegg/unique_rand_labelx_num.txt',header=None)
#count_set = [i[0] for i in np.array(count_setx)]
count_set = [0]
for test_indel in range(1,11): ################## three fold cross validation    ## for KEGG and Reactiome 3 fold CV              #for KEGG and Reactiome 3 fold CV
    X_data_test = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_X_data_array.npy')
    Y_data_test = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_Y_data_array.npy')
    #gene_pair_index_test = np.load('/home/yey3/spatial_nn/processed_data/new_split/rand_1_10fold/'+str(test_indel)+'_test_gene_pair_list_array.npy')
    count_setz = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_gene_pair_index_array.npy')
    #(x_train, y_train,count_set_train) = load_data_TF2(train_TF,data_train)
    y_predictyz = np.load(current_path+'/'+str(test_indel)+'_Ycv_LR_as_nega_rg_5-7_lr_1-6_e100' + '/end_y_predict.npy')
    y_testyz = np.load(current_path+'/'+str(test_indel)+'_Ycv_LR_as_nega_rg_5-7_lr_1-6_e100'  + '/end_y_test.npy')
    y_testy = np.concatenate((y_testy,y_testyz),axis = 0)
    y_predicty = np.concatenate((y_predicty, y_predictyz), axis=0)
    count_set = count_set + [i + count_set[-1] if len(count_set)>0 else i for i in count_setz[1:]]
    ############




print (len(count_set))
###############whole performance
#y_test = y_testy
#y_predict = y_predicty
AUC_set =[]
s = open(save_dir+'/whole_RPKM_AUCs1+2.txt','w')
tprs = []
mean_fpr = np.linspace(0, 1, 100)
#######################################
##################################
fig = plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1])
total_pair = 0
total_auc = 0
print (y_predicty.shape)
    ############
for jj in range(len(count_set)-1):#len(count_set)-1):
    if count_set[jj] < count_set[jj+1]:
        print (test_indel,jj,count_set[jj],count_set[jj+1])
        current_pair = count_set[jj+1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test = y_testy[count_set[jj]:count_set[jj+1]]
        y_predict = y_predicty[count_set[jj]:count_set[jj+1],0]
        # Score trained model.
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)		
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
            # Print ROC curve
        plt.plot(fpr, tpr, color='0.5', lw=0.001,alpha=.2)
        auc = np.trapz(tpr, fpr)
        s.write(str(jj)+'\t'+str(count_set[jj])+'\t'+str(count_set[jj+1])+'\t'+str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)
        total_auc = total_auc + auc * current_pair

mean_tpr = np.median(tprs, axis=0)
mean_tpr[-1] = 1.0
per_tpr = np.percentile(tprs,[40,50,60],axis=0)
mean_auc = np.trapz(mean_tpr,mean_fpr)
plt.plot(mean_fpr, mean_tpr,'k',lw=3,label = 'median ROC')
plt.title("{:.4f}".format(mean_auc),fontsize=15)
plt.fill_between(mean_fpr, per_tpr[0,:], per_tpr[2,:], color='g', alpha=.2,label='quantile')
plt.plot(mean_fpr, per_tpr[0,:],'g',lw=3,alpha=.2)
plt.legend(loc='lower right',fontsize=15)
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.xlabel('FP', fontsize=15)
plt.ylabel('TP', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(save_dir+'/whole_kegg_ROCs1+2_percentile.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.hist(AUC_set,bins = 50)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_hist.pdf')
del fig
s.close()
fig = plt.figure(figsize=(3, 3))
plt.boxplot(AUC_set)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_box.pdf')
del fig
############################

###################################################################### PR
##################################
AUC_set =[]
s = open(save_dir+'/whole_RPKM_AUCs1+2_PR.txt','w')
tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig = plt.figure(figsize=(5, 5))
total_pair = 0
total_auc = 0
print (y_predicty.shape)
    ############
for jj in range(len(count_set)-1):#len(count_set)-1):
    if count_set[jj] < count_set[jj+1]:
        print (test_indel,jj,count_set[jj],count_set[jj+1])
        current_pair = count_set[jj+1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test = y_testy[count_set[jj]:count_set[jj+1]]
        y_predict = y_predicty[count_set[jj]:count_set[jj+1]]
        # Score trained model.
        tpr, fpr, thresholds = metrics.precision_recall_curve(y_test, y_predict)  # , pos_label=1)
        tpr = np.flip(tpr)
        fpr = np.flip(fpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, color='0.5', lw=0.001,alpha=.2)
        auc = np.trapz(tpr, fpr)
        s.write(str(jj)+'\t'+str(count_set[jj])+'\t'+str(count_set[jj+1])+'\t'+str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)
        total_auc = total_auc + auc * current_pair

mean_tpr = np.median(tprs, axis=0)

per_tpr = np.percentile(tprs,[40,50,60],axis=0)
mean_auc = np.trapz(mean_tpr,mean_fpr)
plt.plot(mean_fpr, mean_tpr,'k',lw=3,label = 'median ROC')
plt.title("{:.4f}".format(mean_auc),fontsize=15)
plt.fill_between(mean_fpr, per_tpr[0,:], per_tpr[2,:], color='g', alpha=.2,label='quantile')
plt.plot(mean_fpr, per_tpr[0,:],'g',lw=3,alpha=.2)
plt.legend(loc='lower right',fontsize=15)
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.xlabel('FP', fontsize=15)
plt.ylabel('TP', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(save_dir +'/whole_kegg_ROCs1+2_percentile_PR.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.hist(AUC_set,bins = 50)
plt.savefig(save_dir  +'/whole_kegg_ROCs1+2_hist_PR.pdf')
del fig
s.close()
fig = plt.figure(figsize=(3, 3))
plt.boxplot(AUC_set)
plt.savefig(save_dir  +'/whole_kegg_ROCs1+2_box_PR.pdf')
del fig

from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.datasets import mnist
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import normalized_laplacian
from keras.utils import plot_model
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
current_path = os.path.abspath('.')
# Parameters
l2_reg = 5e-6  # Regularization rate for l2
learning_rate = 1*1e-4  # Learning rate for SGD
batch_size = 32  # Batch size
epochs = 100  # Number of training epochs
es_patience = 50  # Patience fot early stopping

# Load data
import numpy as np

from scipy import sparse
import pickle
with open(current_path+'/seqfish_plus/whole_FOV_distance_I_N_crs_140', 'rb') as fp:
    adj = pickle.load( fp)

# adj = np.load('/home/yey3/spatial_nn/processed_data/sourcedata/sy/FOV_0_distance_I_N_crs.npy')
for test_indel in range(1,11): ################## ten fold cross validation
    X_data_train = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_X_data_array.npy')
    Y_data_train = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_Y_data_array.npy')
    gene_pair_index_train = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_gene_pair_list_array.npy')
    count_setx_train = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_train_gene_pair_index_array.npy')
    X_data_test = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_X_data_array.npy')
    Y_data_test = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_Y_data_array.npy')
    gene_pair_index_test = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_gene_pair_list_array.npy')
    count_set = np.load(current_path+'/rand_1_10fold/'+str(test_indel)+'_test_gene_pair_index_array.npy')
    trainX_index = [i for i in range(Y_data_train.shape[0])]
    validation_index = trainX_index[:int(np.ceil(0.2*len(trainX_index)))]
    train_index = trainX_index[int(np.ceil(0.2*len(trainX_index))):]
    X_train, y_train = X_data_train[train_index],Y_data_train[train_index][:,np.newaxis]
    X_val, y_val= X_data_train[validation_index],Y_data_train[validation_index][:,np.newaxis]
    X_test, y_test= X_data_test,Y_data_test[:,np.newaxis]

    # X_train, y_train, X_val, y_val, X_test, y_test, adj = mnist.load_data()
    # X_train, X_val, X_test = X_train[..., None], X_val[..., None], X_test[..., None]
    N = X_train.shape[-2]  # Number of nodes in the graphs
    F = X_train.shape[-1]  # Node features dimensionality
    n_out = y_train.shape[-1]  # Dimension of the target

    fltr = normalized_laplacian(adj)

    # Model definition
    X_in = Input(shape=(N, F))
    # Pass A as a fixed tensor, otherwise Keras will complain about inputs of
    # different rank.
    A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

    graph_conv = GraphConv(32,activation='elu',kernel_regularizer=l2(l2_reg),use_bias=True)([X_in, A_in])
    graph_conv = GraphConv(32,activation='elu',kernel_regularizer=l2(l2_reg),use_bias=True)([graph_conv, A_in])
    #graph_conv = GraphConv(32,activation='elu',kernel_regularizer=l2(l2_reg),use_bias=True)([graph_conv, A_in])
    flatten = Flatten()(graph_conv)
    fc = Dense(512, activation='relu')(flatten)
    output = Dense(n_out, activation='sigmoid')(fc)

    # Build model
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
    model.summary()

    plot_model(model, to_file='gcn_LR_spatial_1.png', show_shapes=True)
    save_dir = current_path+'/'+str(test_indel)+'_Ycv_LR_as_nega_rg_5-7_lr_1-6_e'+str(epochs)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    early_stopping = EarlyStopping(monitor='val_acc', patience=600, verbose=0, mode='auto')
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_acc', verbose=1,save_best_only=True, mode='auto', period=1)
    callbacks = [checkpoint2, early_stopping]

    # Train model
    validation_data = (X_val, y_val)
    model.load_weights(save_dir+'/gcn_LR_1.h5')
    #history = model.fit(X_train,y_train,batch_size=batch_size,validation_data=validation_data,epochs=epochs,callbacks=callbacks)

    # Load best model
    # Save model and weights

    #model_name = 'gcn_LR_1.h5'
    #model_path = os.path.join(save_dir, model_name)
    #model.save(model_path)
    #print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(X_test, y_test, verbose=1,batch_size=batch_size)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    y_predict = model.predict(X_test)
    np.save(save_dir + '/min_y_test.npy', y_test)
    np.save(save_dir + '/min_y_predict.npy', y_predict)
    ############################################################################## plot training process
    #######################################

    #############################################################
    #########################
    y_testy = y_test
    y_predicty = y_predict
    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    # plt.grid()
    AUC_set = []
    s = open(save_dir + '/divided_interaction.txt', 'w')
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)  # 3068
    for jj in range(len(count_set) - 1):  # len(count_set)-1):
        if count_set[jj] < count_set[jj + 1]:
            print(test_indel, jj, count_set[jj], count_set[jj + 1])
            y_test = y_testy[count_set[jj]:count_set[jj + 1]]
            y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
            # Score trained model.
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # Print ROC curve
            plt.plot(fpr, tpr, color='0.5', lw=0.001, alpha=.2)
            auc = np.trapz(tpr, fpr)
            s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
            print('AUC:', auc)
            AUC_set.append(auc)

    mean_tpr = np.median(tprs, axis=0)
    mean_tpr[-1] = 1.0
    per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
    mean_auc = np.trapz(mean_tpr, mean_fpr)
    plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
    plt.title(str(mean_auc))
    plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
    plt.plot(mean_fpr, per_tpr[0, :], 'g', lw=3, alpha=.2)
    plt.legend(loc='lower right')
    plt.savefig(save_dir + '/min_divided_interaction_percentile.pdf')
    del fig
    fig = plt.figure(figsize=(5, 5))
    plt.hist(AUC_set, bins=50)
    plt.savefig(save_dir + '/min_divided_interaction_hist.pdf')
    del fig
    s.close()


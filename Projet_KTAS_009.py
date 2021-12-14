# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:00:31 2021

@author: Guillaume
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
import csv
import os
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import statsmodels.api as sm
import time
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

import warnings
warnings.filterwarnings("ignore")

os.chdir(".")

def get_np_from_csv(file_adress, dtype = None):
    csv_file = open(file_adress)

    csv_reader_temp = csv.reader(csv_file, delimiter = ';')

    data_raw = []
    for row in csv_reader_temp:
        data_raw.append(row)
    data_raw_np = np.array(data_raw, dtype = np.dtype('i4'))
    #print(data_raw)
    return data_raw_np



def remove_col_from_pd(pd_df, col_names):
    data_raw = pd_df
    for i in range(len(col_names)):
        data_raw = data_raw.drop([col_names[i]],axis = 1)
    return data_raw


def create_dataset_with_objective(pd_df, objective_name):
    data = {}
    data['target'] = pd_df[objective_name]
    data['data'] = pd_df.drop(objective_name, axis = 1)
    return data


def split_dataset(pd_df, prop = 0.5, seed = None):
    n_obs = len(pd_df)
    np.random.RandomState(seed)
    np.random.seed(seed)
    n_group_1 = int(n_obs * prop)
    train_index = np.random.choice(np.arange(n_obs), n_group_1, replace = False)
    train_index.sort()
    data_group_1 = pd_df.iloc[train_index,:]
    test_mask = np.full(n_obs, True, dtype = bool)
    test_mask[train_index] = False
    data_group_2 = pd_df.iloc[test_mask,:]
    return data_group_1, data_group_2

def split_dataset_train_val_test(pd_df, val_test = (0.2,0.2), seed = None):
    df_train_val, df_test = split_dataset(pd_df, prop = 1 - val_test[1], seed = seed)
    df_train, df_val = split_dataset(df_train_val, prop = (1 - val_test[0] - val_test[0])/(1 - val_test[1]), seed = seed)
    return df_train, df_val, df_test


        
def random_hp_search_xgb(n_try, max_iter, dtrain, dval):
    possible_parameters = np.mgrid[2:15.1:1, 0:3.1:0.2, 1:50.1:5].reshape(3,-1).T
    
    recorded_hps = []
    
    time_sample = 0
    time_train = 0
    best_error = 9e9
    
    for round_temp in range(n_try):
        print("doing time : " + str(round_temp))
        time_sample -= time.time()
        hp_temp = possible_parameters[np.random.randint(0,len(possible_parameters) - 1)]
        time_sample += time.time()

        time_train -= time.time()
        params = {'max_depth': int(hp_temp[0]), 'eta': 0.1 ** hp_temp[1], 'min_child_weight': int(hp_temp[2])}
        progress = dict()
        bst = xgb.train(params, dtrain, max_iter, [(dtrain, 'train'), (dval, 'val')], evals_result = progress, verbose_eval = False)
        time_train += time.time()
        
        val_error = progress['val']['rmse'][np.argmin(progress['val']['rmse'])]
        temp_hps = [hp_temp, val_error]
        recorded_hps.append(temp_hps)
        
        if val_error < best_error:
            best_error = val_error
            best_hp = hp_temp

        print("Changement random : " + str(round_temp))
        print("Trying params : " + str(hp_temp))
        print('best error yet : ' + str(best_error))
        print('Training time : ' + str(time_train))
        print('Sampling time : ' + str(time_sample))
        
        
    return best_hp, best_error, recorded_hps

def random_hp_search_sk_xgb(n_try, max_iter, dtrain, dval):
    possible_parameters = np.mgrid[2:15.1:1, 0:3.1:0.2, 1:50.1:5].reshape(3,-1).T
    
    recorded_hps = []
    
    time_sample = 0
    time_train = 0
    best_error = 9e9
    
    for round_temp in range(n_try):
        print("doing time : " + str(round_temp))
        time_sample -= time.time()
        hp_temp = possible_parameters[np.random.randint(0,len(possible_parameters) - 1)]
        time_sample += time.time()

        time_train -= time.time()
        params = {'max_depth': int(hp_temp[0]), 'eta': 0.1 ** hp_temp[1], 'min_child_weight': int(hp_temp[2])}
        progress = dict()
        
        params = {'max_depth': 5, 'eta': 0.1 ** 1, 'min_child_weight': 5, 'n_estimators': 200}
        model = XGBClassifier(max_depth = int(hp_temp[0]), learning_rate = 0.1 ** hp_temp[1], min_child_weight = int(hp_temp[2]), n_estimators = 5000)
        model.fit(dtrain['data'], dtrain['target'], 
                  eval_set = [(dtrain['data'], dtrain['target']), (dval['data'], dval['target'])],
                  early_stopping_rounds = 200, 
                  eval_metric = 'mlogloss', verbose = False)
        #model.best_iteration
        #y_pred_tr = model.predict(data_tr['data'])
        y_pred_vl = model.predict(dval['data'])
        val_error = 1 - np.mean(dval['target'] == y_pred_vl)
        
        
        #model = XGBClassifier()
        #model.fit(params, dtrain, max_iter, [(dtrain, 'train'), (dval, 'val')], evals_result = progress, verbose_eval = False)
        time_train += time.time()
        
        #val_error = progress['val']['rmse'][np.argmin(progress['val']['rmse'])]
        temp_hps = [hp_temp, val_error]
        recorded_hps.append(temp_hps)
        
        if val_error < best_error:
            best_error = val_error
            best_hp = hp_temp

        print("Changement random : " + str(round_temp))
        print("Trying params : " + str(hp_temp))
        print('best error yet : ' + str(best_error))
        print('Training time : ' + str(time_train))
        print('Sampling time : ' + str(time_sample))
        
        
    return best_hp, best_error, recorded_hps


def full_hp_search_cart(max_iter, dtrain, dval, random = False, n_try = 1):
    possible_parameters = np.mgrid[1:100.1:1, 1:15.1:1, 0:1.1:0.01].reshape(3, -1).T
    
    recorded_hps = []
    
    time_sample = 0
    time_train = 0
    best_error = 9e9
    best_score = 0
    
    if random:
        n_rep = n_try
    else:
        n_rep = len(possible_parameters)
        
    #for round_temp in range(n_try):
    for round_temp in range(n_rep):
        #print("doing time : " + str(round_temp))
        time_sample -= time.time()
        if random:
            hp_temp = possible_parameters[np.random.randint(0,len(possible_parameters) - 1)]
        else:
            hp_temp = possible_parameters[round_temp]
        #hp_temp = possible_parameters[np.random.randint(0,len(possible_parameters) - 1)]
        time_sample += time.time()

        time_train -= time.time()
        params = {'max_depth': int(hp_temp[0]), 'min_samples_leaf': int(hp_temp[1]), 'min_impurity_decrease': hp_temp[2]}
        #cart_tree = DecisionTreeClassifier(params)
        cart_tree = DecisionTreeClassifier(max_depth = int(hp_temp[0]), min_samples_leaf = int(hp_temp[1]), min_impurity_decrease = hp_temp[2])
        cart_tree.fit(dtrain['data'], dtrain['target'])
        time_train += time.time()
        
        val_score = cart_tree.score(dval['data'], dval['target'])
        temp_hps = [hp_temp, val_score]
        recorded_hps.append(temp_hps)
        #print(val_error)
        
        if val_score > best_score:
            best_score = val_score
            best_hp = hp_temp

        if round_temp % 100 == 0:
            print("Changement random : " + str(round_temp) + ' sur ' + str(n_rep))
            print("Trying params : " + str(hp_temp))
            print('best error yet : ' + str(best_score))
            print('Training time : ' + str(time_train))
            print('Sampling time : ' + str(time_sample))
        
        
    return best_hp, best_score, recorded_hps


def full_hp_search_randomforest(max_iter, dtrain, dval, random = False, n_try = 1):
    possible_parameters = np.mgrid[1:1000.20:1, 1:200.5:1, 1:15.1:1, 0:1.1:0.01].reshape(4, -1).T
    
    recorded_hps = []
    
    time_sample = 0
    time_train = 0
    best_error = 9e9
    best_score = 0
    
    if random:
        n_rep = n_try
    else:
        n_rep = len(possible_parameters)
        
    #for round_temp in range(n_try):
    for round_temp in range(n_rep):
        #print("doing time : " + str(round_temp))
        time_sample -= time.time()
        if random:
            hp_temp = possible_parameters[np.random.randint(0,len(possible_parameters) - 1)]
        else:
            hp_temp = possible_parameters[round_temp]
        #hp_temp = possible_parameters[np.random.randint(0,len(possible_parameters) - 1)]
        time_sample += time.time()

        time_train -= time.time()
        params = {'n_estimators': int(hp_temp[0]), 'max_depth': int(hp_temp[1]), 'min_samples_leaf': int(hp_temp[2]), 'min_impurity_decrease': hp_temp[3]}
        #cart_tree = DecisionTreeClassifier(params)
        random_forest = RandomForestClassifier(n_estimators = int(hp_temp[0]), max_depth = int(hp_temp[1]), min_samples_leaf = int(hp_temp[2]), min_impurity_decrease = hp_temp[3])
        random_forest.fit(dtrain['data'], dtrain['target'])
        time_train += time.time()
        
        val_score = random_forest.score(dval['data'], dval['target'])
        temp_hps = [hp_temp, val_score]
        recorded_hps.append(temp_hps)
        #print(val_error)
        
        if val_score > best_score:
            best_score = val_score
            best_hp = hp_temp

        if round_temp % 100 == 0:
            print("Changement random : " + str(round_temp) + ' sur ' + str(n_rep))
            print("Trying params : " + str(hp_temp))
            print('best error yet : ' + str(best_score))
            print('Training time : ' + str(time_train))
            print('Sampling time : ' + str(time_sample))
        
        
    return best_hp, best_score, recorded_hps




def get_coef_p_values(x, y):
    X = x #data_train_NRS_pain.astype(float)
    y = y #data_train_NRS_pain_target.astype(float)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    summary = est2.summary()
    results_as_html = summary.tables[1].as_html()
    summary_pd = pd.read_html(results_as_html, header=0, index_col=0)[0]
    summary_pd = summary_pd.dropna()
    return summary_pd

def backward_p_value(x, y, p_tresh = 0.05):
    doing_back = True
    while doing_back:
        summary = get_coef_p_values(x, y)
        summary = summary.loc[summary.index != 'const',:]
        worst_index = np.argmax(summary['P>|t|'])
        print(summary)
        #doing_back = False
        print(summary['P>|t|'][worst_index])
        if summary['P>|t|'][worst_index] > p_tresh:
            #print(worst_index)
            print(summary.index[worst_index])
            #print(x)
            #print(list(x.columns))
            x = x.drop(summary.index[worst_index], 1)
        else:
            doing_back = False
        #time.sleep(0)
    return x


#########################################################
#Prétraitement des données
#########################################################
data_raw_pd = pd.read_csv('data_text_04.csv', delimiter = ";")

columns_to_drop = ['Chief_complain','KTAS_RN','Diagnosis in ED', 'Disposition', 'Length of stay_min', 'KTAS duration_min', 'mistriage', 'Arrival mode', 'Error_group']
data_raw = remove_col_from_pd(data_raw_pd, columns_to_drop)

seed = 0
prop = 0.5

data_raw['NRS_pain'][data_raw['NRS_pain'] == '#BOÞ!'] = 0
data_raw['Saturation'][data_raw['Saturation'].isnull()] = 100

data_raw_copy = copy.deepcopy(data_raw)
data_raw_copy = data_raw_copy.dropna()
full_index = np.logical_and(np.logical_and(np.logical_and(np.logical_and(data_raw_copy['SBP'] != '??', 
                                                           data_raw_copy['DBP'] != '??'),
                                                            data_raw_copy['RR'] != '??'),
                                                            data_raw_copy['BT'] != '??'),
                                                            data_raw_copy['Saturation'] != '??')
data_raw_copy = data_raw_copy.loc[full_index, :]
data_raw_copy = data_raw_copy.astype(float)
#for i in range(len(np.sum(data_raw_copy == '??'))):
#    if i < 24:
#        print(i, np.sum(data_raw_copy == '??')[i])

data_x = data_raw_copy.loc[:, data_raw_copy.columns != 'KTAS_expert']
data_y = data_raw_copy.loc[:, data_raw_copy.columns == 'KTAS_expert']

#########################################################
#Fin pré-traitement des données
#########################################################

###############################################
#Réduction de dimension
###############################################


def plot_clustering(X_red, labels, title, savepath):
    # Tiré de https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html
    # Auteur : Gael Varoquaux
    # Distribué sous license BSD
    #
    # - X_red: array numpy contenant les caractéristiques (features)
    #   des données d'entrée, réduit à 2 dimensions / numpy array containing
    #   the features of the input data of the input data, reduced to 2 dimensions
    #
    # - labels: un array numpy contenant les étiquettes de chacun des
    #   éléments de X_red, dans le même ordre. / a numpy array containing the
    #   labels of each of the elements of X_red, in the same order.
    #
    # - title: le titre que vous souhaitez donner à la figure / the title you want
    #   to give to the figure
    #
    # - savepath: le nom du fichier où la figure doit être sauvegardée / the name
    #   of the file where the figure should be saved
    #
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(9, 6), dpi=160)
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(labels[labels.index[i]]),
                color=plt.cm.nipy_spectral(labels[labels.index[i]] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

def reduce_data(reducer, data, targets, columns_to_not_reduce = [], title = 'Reduced data', n_reduced_var = 2):
    data_to_reduce = remove_col_from_pd(data, columns_to_not_reduce)
    data_reduced = reducer.fit_transform(data_to_reduce)
    # plot_clustering(data_reduced, targets, title, False)
    new_data = data_x[columns_to_not_reduce]
    for i in range(n_reduced_var):
        new_data.loc[:,'Reduced variable ' + str(i + 1)] = data_reduced[:,i]
    return new_data

columns_to_not_reduce = ['Group','Sex','Age', 'Patients number per hour', 'Injury', 'Mental', 'Pain', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']

PCA_reducer = PCA(2)
MDS_reducer = MDS(2, n_init = 1)
TSNE_reducer = TSNE(2)

new_data_PCA = reduce_data(PCA_reducer, data_x, data_y['KTAS_expert'], columns_to_not_reduce, title = 'Reduced data by PCA', n_reduced_var = 2)
new_data_MDS = reduce_data(MDS_reducer, data_x, data_y['KTAS_expert'], columns_to_not_reduce, title = 'Reduced data by MDS', n_reduced_var = 2)
new_data_TSNE = reduce_data(TSNE_reducer, data_x, data_y['KTAS_expert'], columns_to_not_reduce, title = 'Reduced data by TSNE', n_reduced_var = 2)


#########################################################
#XGBoost
#########################################################
def create_dataset_with_objective_no_pd(x,y):
    data = {}
    data['data'] = x
    data['target'] = y
    return data
    
def split_dataset_2(data, target_name, val_test = (0.2,0.2), seed = 0):
    df_train, df_val, df_test = split_dataset_train_val_test(data, val_test = val_test, seed = seed)
    data_tr = create_dataset_with_objective(df_train, target_name)
    data_vl = create_dataset_with_objective(df_val, target_name)
    data_ts = create_dataset_with_objective(df_test, target_name)
    return data_tr, data_vl, data_ts

data_tr, data_vl, data_ts = split_dataset_2(data_raw_copy, 'KTAS_expert', seed = 42)
data_PCA = new_data_PCA
data_PCA['KTAS_expert'] = data_y['KTAS_expert']
data_tr_PCA, data_vl_PCA, data_ts_PCA = split_dataset_2(new_data_PCA, 'KTAS_expert', seed = 42)
print(data_tr_PCA)
data_MDS = new_data_MDS
data_MDS['KTAS_expert'] = data_y['KTAS_expert']
data_tr_MDS, data_vl_MDS, data_ts_MDS = split_dataset_2(new_data_MDS, 'KTAS_expert', seed = 42)
data_TSNE = new_data_TSNE
data_TSNE['KTAS_expert'] = data_y['KTAS_expert']
data_tr_TSNE, data_vl_TSNE, data_ts_TSNE = split_dataset_2(new_data_TSNE, 'KTAS_expert', seed = 42)

# dtrain = xgb.DMatrix(data_tr['data'], data_tr['target'])
# dval = xgb.DMatrix(data_vl['data'], data_vl['target'])
# dtest = xgb.DMatrix(data_ts['data'], data_ts['target'])


# print(np.mean(data_raw['KTAS_expert'] == data_raw_pd['KTAS_RN']))
# n_try = 50
# max_iter = 5000
# seed = 0
# np.random.RandomState(seed)
# np.random.seed(seed)
# best_hp_cart, val_error_cart, recorded_hps_cart = full_hp_search_cart(max_iter, data_tr, data_vl, random = True, n_try = 10000)
# best_hp_cart_PCA, val_error_cart_PCA, recorded_hps_cart_PCA = full_hp_search_cart(max_iter, data_tr_PCA, data_vl_PCA, random = True, n_try = 10000)
# best_hp_cart_MDS, val_error_cart_MDS, recorded_hps_cart_MDS = full_hp_search_cart(max_iter, data_tr_MDS, data_vl_MDS, random = True, n_try = 10000)
# best_hp_cart_TSNE, val_error_cart_TSNE, recorded_hps_cart_TSNE = full_hp_search_cart(max_iter, data_tr_TSNE, data_vl_TSNE, random = True, n_try = 10000)

# n_try = 200 #200
# max_iter = 5000
# best_hp, val_error, recorded_hps = random_hp_search_sk_xgb(n_try, max_iter, {'data' : data_tr['data'], 'target' : data_tr['target']}, {'data' : data_vl['data'], 'target' : data_vl['target']})
# best_hp_PCA, val_error_PCA, recorded_hps_PCA = random_hp_search_sk_xgb(n_try, max_iter, {'data' : data_tr_PCA['data'], 'target' : data_tr_PCA['target']}, {'data' : data_vl_PCA['data'], 'target' : data_vl_PCA['target']})
# best_hp_MDS, val_error_MDS, recorded_hps_MDS = random_hp_search_sk_xgb(n_try, max_iter, {'data' : data_tr_MDS['data'], 'target' : data_tr_MDS['target']}, {'data' : data_vl_MDS['data'], 'target' : data_vl_MDS['target']})
# best_hp_TSNE, val_error_TSNE, recorded_hps_TSNE = random_hp_search_sk_xgb(n_try, max_iter, {'data' : data_tr_TSNE['data'], 'target' : data_tr_TSNE['target']}, {'data' : data_vl_TSNE['data'], 'target' : data_vl_TSNE['target']})

# n_try = 50
# max_iter = 5000
# seed = 0
# np.random.RandomState(seed)
# np.random.seed(seed)
# best_hp_cart, val_error_cart, recorded_hps_cart = full_hp_search_randomforest(max_iter, data_tr, data_vl, random = True, n_try = 10000)
# best_hp_cart_PCA, val_error_cart_PCA, recorded_hps_cart_PCA = full_hp_search_randomforest(max_iter, data_tr_PCA, data_vl_PCA, random = True, n_try = 10000)
# best_hp_cart_MDS, val_error_cart_MDS, recorded_hps_cart_MDS = full_hp_search_randomforest(max_iter, data_tr_MDS, data_vl_MDS, random = True, n_try = 10000)
# best_hp_cart_TSNE, val_error_cart_TSNE, recorded_hps_cart_TSNE = full_hp_search_randomforest(max_iter, data_tr_TSNE, data_vl_TSNE, random = True, n_try = 10000)







# ##############################################
# #Random stuff
# ##############################################




# #array([4. , 2.2, 1. ])
# hp_temp = np.array([4. , 2.2, 1. ])
# model = XGBClassifier(max_depth = int(hp_temp[0]), learning_rate = 0.1 ** hp_temp[1], min_child_weight = int(hp_temp[2]), n_estimators = 5000)
# model.fit(data_tr['data'], data_tr['target'], 
#                   eval_set = [(data_tr['data'], data_tr['target']), (data_vl['data'], data_vl['target'])],
#                   early_stopping_rounds = 200, 
#                   eval_metric = 'mlogloss', verbose = False)
# y_pred_xgb_vl = model.predict(data_vl['data'])
# confusion_mat_xgb = np.zeros((5,5))
# for i in range(len(y_pred_xgb_vl)):
#     confusion_mat_xgb[int(data_vl['target'].loc[data_vl['target'].index[i]]) - 1, int(y_pred_xgb_vl[i]) - 1] += 1
# confusion_mat_xgb_2 = confusion_mat_xgb / np.sum(confusion_mat_xgb)
# np.mean(np.array(data_vl['target']) == np.array(y_pred_xgb_vl))

# rounded_imp = np.round(model.feature_importances_ * 100)
# max_imp = int(np.max(rounded_imp))
# for i in range(max_imp):
#     print(max_imp - i, data_tr['data'].columns[rounded_imp == max_imp - i])


# y_pred_xgb_ts = model.predict(data_ts['data'])
# np.mean(np.array(data_ts['target']) == np.array(y_pred_xgb_ts))


# data_y
# data_y_rn = data_raw_pd.loc[data_y.index, data_raw_pd.columns == 'KTAS_RN']
# len(data_y)
# confusion_mat = np.zeros((5,5))
# for i in range(len(data_y)):
#     confusion_mat[int(data_y.loc[data_y.index[i]]) - 1, data_y_rn.loc[data_y.index[i]] - 1] += 1
# confusion_mat_2 = confusion_mat / np.sum(confusion_mat)
# np.mean(np.array(data_y) == np.array(data_y_rn))




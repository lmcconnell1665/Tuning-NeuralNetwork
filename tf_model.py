"""
Tuning-NeuralNetwork
10/4/2020
Dunn, McGinnis, McConnell
"""

import tensorflow as tf
import pandas as pd
import datetime
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from random import sample
import numpy as np  
import json

# tf.__version__

#####################
#### DATA IMPORT ####
#####################

# Import data
pricing = pd.read_csv('pricing.csv', sep = ';')
pricing = pricing.sample(10000)

# Pull on y variable
y = pricing['gross_margin'] / pricing['adjusted_duration_seconds_sum']

# Keep only relavant x variables
X = pricing[['master_id',
                   'unit_cost',
                   'host_full_name_1_array',
                   'show_brand_label_1_array',
                   'show_type_array',
                   'showing_start_date_time_min',
                   'merch_department',
                   'merch_class_name',
                   'country_of_origin',
                   'unit_offer_price']]

# Create training and holdout samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

################################
#### SPARSING - COMING SOON ####
################################

# Sparse encoding categorical variables
# def sparse_tensor(X_cat_int):
#     X_cat_sparse = []
#     for i in range(len(X_cat_int)):
#         indices = [0,X_cat_int[i]]
#         X_cat_sparse.append(tf.sparse.SparseTensor(indices = [indices], 
#                                                     values = [1], 
#                                                     dense_shape = [1,len(np.unique(X_cat_int))]))
#     return X_cat_sparse

# X_host_full_name_1_array_sparse_train = np.array(sparse_tensor(X_train['host_full_name_1_array'].values))
# X_host_full_name_1_array_sparse_test = np.array(sparse_tensor(X_test['host_full_name_1_array'].values))

#####################
#### TF FUNCTION ####
#####################

def tune_parameters(batch_size_entry = 1,
                    hidden_node_entry = 2,
                    hidden_layers_entry = 2,
                    activation_func_entry = 'relu',
                    optimizer_func_entry = 'Adam',
                    epoch_entry = 10):
    
    #Start timer
    start = datetime.datetime.now()

    #Specify architecture
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1],), name='input')
    
    if hidden_layers_entry == 1:
        hidden1 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden1')(inputs)
        output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden1)
    elif hidden_layers_entry == 2:
        hidden1 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden2')(hidden1)
        output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden2)
    elif hidden_layers_entry == 3:
        hidden1 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden2')(hidden1)
        hidden3 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden3')(hidden2)
        output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden3)
    elif hidden_layers_entry == 4:
        hidden1 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden2')(hidden1)
        hidden3 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden3')(hidden2)
        hidden4 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden4')(hidden3)
        output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden4)
    elif hidden_layers_entry == 5:
        hidden1 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden2')(hidden1)
        hidden3 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden3')(hidden2)
        hidden4 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden4')(hidden3)
        hidden5 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden5')(hidden4)
        output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden5)

    #Create model 
    model = tf.keras.Model(inputs = inputs, outputs = output)

    #Compile model
    if optimizer_func_entry == "SGD":
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(lr = 0.001))
    elif optimizer_func_entry == "Adam":
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
                                                                         beta_1 = 0.9,
                                                                         beta_2 = 0.999,
                                                                         epsilon = 1e-07))

    #Fit model
    model.fit(x=X_train,y=y_train, batch_size=batch_size_entry, epochs=epoch_entry)

    #making a prediction (first record)
    yhat = model.predict(x=X_test[0:1])

    #making a prediction (all records)
    yhat = model.predict(x=X_test)
    
    #Stop timer
    time = datetime.datetime.now() - start
    
    #getting the loss (on for example a test set)
    loss = model.evaluate(X_test,y_test)
    
    #Gather results
    results = {'time': str(time),
               'batch_size': batch_size_entry,
               'hidden_node_count': hidden_node_entry,
               'hidden_layers_count': hidden_layers_entry,
               'activation_function': activation_func_entry,
               'optimizer': optimizer_func_entry,
               'epoch_count': epoch_entry,
               'loss': loss}

    return results

#####################
###### TUNING #######
#####################

tune_grid_results = list()

try_batch_size = list([1, 10, 50, 100, 1000])
try_hidden_node = list([2, 5])
try_hidden_layers = list([1, 3, 5]) #can only handle 1 through 5
try_activation_func = ['relu', 'sigmoid']
try_optimizer_func = ['SGD', 'Adam']
try_epoch_size = list([10, 20, 30])

for i in range(len(try_batch_size)):
    try_this_batch_size = try_batch_size[i]
    
    for j in range(len(try_hidden_node)):
        try_this_hidden_node = try_hidden_node[j]
        
        for k in range(len(try_activation_func)):
            try_this_activation_func = try_activation_func[k]
            
            for l in range(len(try_optimizer_func)):
                try_this_optimizer_func = try_optimizer_func[l]
                
                for m in range(len(try_hidden_layers)):
                    try_this_hidden_layer = try_hidden_layers[m]
                    
                    for n in range(len(try_epoch_size)):
                        try_this_epoch_size = try_epoch_size[n]
    
                        tune_grid_results.append(tune_parameters(batch_size_entry = try_this_batch_size,
                                        hidden_node_entry = try_this_hidden_node,
                                        hidden_layers_entry = try_this_hidden_layer,
                                        activation_func_entry = try_this_activation_func,
                                        optimizer_func_entry = try_this_optimizer_func,
                                        epoch_entry = try_this_epoch_size))

#########################
#### SAVE RESULTS #######
#########################       

with open('outputfile', 'w') as fout:
    json.dump(tune_grid_results, fout)

#####################
#### OLD CODE #######
#####################

#number of iterations completed
# len(tune_grid_results)

#find the lowest loss
# seq = [x['loss'] for x in tune_grid_results]
# next(item for item in tune_grid_results if item["loss"] == min(seq))

#summarizing the model
# model.summary()

# #getting the weights
# model.layers
# hidden1weights = model.layers[1]
# hidden1weights.get_weights()
    
# Write results to json file each iteration
# file1 = open("results.txt", "a")  # append mode 
#                         file1.write(json.dumps(tune_parameters(batch_size_entry = try_this_batch_size,
#                                         hidden_node_entry = try_this_hidden_node,
#                                         hidden_layers_entry = try_this_hidden_layer,
#                                         activation_func_entry = try_this_activation_func,
#                                         optimizer_func_entry = try_this_optimizer_func,
#                                         epoch_entry = try_this_epoch_size))) 
#                         file1.close()
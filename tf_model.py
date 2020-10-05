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

tf.__version__

#####################
#### DATA IMPORT ####
#####################

# Import data
pricing = pd.read_csv('pricing.csv', sep = ';')
pricing = pricing[0:100]

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

#####################
#### TF FUNCTION ####
#####################

def tune_parameters(batch_size_entry = 1, hidden_node_entry = 2, activation_func_entry = 'relu'):
    
    #Start timer
    start = datetime.datetime.now()

    #Specify architecture
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1],), name='input')

    hidden1 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden1')(inputs)
    hidden2 = tf.keras.layers.Dense(units=hidden_node_entry, activation=activation_func_entry, name='hidden2')(hidden1)
    output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden2)

    #Create model 
    model = tf.keras.Model(inputs = inputs, outputs = output)

    #Compile model
    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(lr = 0.001))

    #Fit model
    model.fit(x=X_train,y=y_train, batch_size=batch_size_entry, epochs=10)

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
               'activation_function': activation_func_entry,
               'loss': loss}

    return results

#####################
###### TUNING #######
#####################

tune_grid_results = list()

try_batch_size = list(range(0,5))
try_hidden_node = list(range(0,5))
try_activation_func = ['relu', 'sigmoid']

for i in range(len(try_batch_size)):
    try_this_batch_size = try_batch_size[i]
    
    for j in range(len(try_hidden_node)):
        try_this_hidden_node = try_hidden_node[j]
        
        for k in range(len(try_activation_func)):
            try_this_activation_func = try_activation_func[k]
    
            tune_grid_results.append( tune_parameters(batch_size_entry = try_this_batch_size,
                            hidden_node_entry = try_this_hidden_node,
                            activation_func_entry = try_this_activation_func))


# Parameters that need to be added to tune: number of hidden layers & optimization function

#####################
#### OLD CODE #######
#####################

#summarizing the model
# model.summary()

# #getting the weights
# model.layers
# hidden1weights = model.layers[1]
# hidden1weights.get_weights()

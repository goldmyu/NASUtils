import pandas as pd
import utils
import numpy as np
import torch
import torch.nn as nn


# region ================================ Experiment description =======================================================
"""
Experiment description and details

We are trying to find the best representation for both topology and hyper-params and how to integrate them together and
feed them as input to our RNN model as a time-series data, we will use an a adjacency's matrix to encode the topology
and a long vector for all the layers HP`s in time step t and will input them both to an RNN based model at each time 
step in order to predict the test-set accuracy of the model.

Topology representation:
    We will represent the topology using and adjacency matrix, this will be used as the RNN first input
    The adj matrix will have values of 0 and 1 to represent a Edge between two nodes
    On the adj matrix diagonal we will have a categorical numeric value to represent the type of the layer
        as of : convolution_layer=1, dropout_layer=2 and so on...

Training Hyper-parameters representation:
    We will represent all Hyper-params (HP) using a large vector - this will be the RNN second input
    Each layer has a vector of hyper-parameters for each training step at time t
    We will concatenate each layer HP vector to a large vector of hyper-param of all the network at time t
    as of: [layer1_(time t)[weight,bias,grad,activ],layer2(time_t)[weigh,bias,grad,activ],...layer_n]

"""
# endregion
# region ================================ Experiment Config ============================================================

data_folder = '../generated_files/experiment_6/'


# endregion
# region ============================= Util Methods ====================================================================

def create_and_populate_adj_matrix(_model_layers_dict):
    matrix_dim = len(_model_layers_dict)
    _adj_matrix = np.zeros([matrix_dim, matrix_dim])

    index = 0
    for layer_num, layer in _model_layers_dict.items():
        layer_type = layer['layer_type']
        layer_type_num = utils.layer_type_enumerate.get(layer_type)
        _adj_matrix[index, index] = layer_type_num
        # TODO - only works for sequential network, no parallel layer, no skip connections
        if index + 1 < len(_model_layers_dict):
            _adj_matrix[index, index + 1] = 1
        index += 1
    return _adj_matrix


# endregion
# region ================================ Load Model ===================================================================

all_models_df = pd.read_csv(data_folder + 'abstract_models.csv')
first_model = all_models_df.iloc[[0]]
model_id = first_model['model_id'][0]
model_layers_str = first_model['model_layers'][0]
model_training_data_df = utils.load_model_training_data(data_folder, model_id)

# Convert model string to json
model_layers_dict = utils.model_layers_str_to_dict(model_layers_str)

# endregion
# region ============================= Create Adj Matrix ===============================================================
# TODO - how should we address identity layers and their connections?
adj_matrix = create_and_populate_adj_matrix(model_layers_dict)
adj_matrix_as_vector = adj_matrix.flatten()

# endregion
# region ============================= Create HP vector ================================================================

# TODO - get training data by iteration and epoch numbers
iteration = 0
epoch = 0
sub = model_training_data_df[
    (model_training_data_df['iteration'] == iteration) & (model_training_data_df['epoch'] == epoch)]

# remove layer_name coloumn and iteration & epoch coloumns
sub_only_data = sub.drop(['layer_name','iteration','epoch'], axis=1)

vector_sub_only_data = sub_only_data.values.flatten()
# endregion
# region ==================== Create training data-set for RNN =========================================================

# Create large vector for all HP and initalize to zeros - size 25x22=550 - 25 hp types and 22 layers
hp_vector = np.zeros(550)

# Iterate over layer_dict and compare each layer name to the layer in the training data base.
# If it is the same layer - insert hp into hp_vector, else skip 25 indexes leaving zeros.
for layer_nunber, layer in model_layers_dict.items():
    print(layer)


# endregion
# region ============================= Create RNN model ================================================================

# Adj matrix as vector is of size 484
# Training meta-data per iteration is of size

input_dim = 5
hidden_dim = 10
n_layers = 1
batch_size = 1
seq_len = 1
lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
inp = torch.randn(batch_size, seq_len, input_dim)
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

# endregion

print('End')






# RNN model with many-to-one arch, because we assume fix length of sequence data
# used LSTM type cell
# RNN network does not yield good performance
# I tried 2 cases,
# One with the two inputs (dual-input RNN) that gets a adj matrix and vector of HP with 0 where we had layers with not params
# Second - single input RNN, adj matrix flattened to vector and concated to the HP vector


# How to evalute?
#  difference in accuracy and AUC between test-set and rnn prediction - run experiment again and also create AUC
# How we Train the RNN - maybe train many-to-many, every step, try to predict the loss on the train batch and in the final step
#  try and predict the test-set accuracy/loss






# def make_model(self, seq_size, features_size, num_of_classes):
#     model = keras.Sequential()
#
#     # model.add(Bidirectional(LSTM(100, input_shape=(sequence_length, features_size), activation='relu',
#     #                              return_sequences=False)))
#
#     model.add(LSTM(128, input_shape=(seq_size, features_size), activation='linear',
#                    return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=keras.regularizers.l2(0.001)))
#     model.add(LeakyReLU(alpha=0.1))
#     # model.add(Bidirectional(LSTM(units=100, return_sequences=False, activation='relu')))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
#     model.add(Dense(num_of_classes, activation="softmax"))
#     adam = optimizers.Adam(lr=0.0001)
#     model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     return model


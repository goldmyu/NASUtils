import ast

import pandas as pd
import utils

# region ================================ Experiment description =======================================================
"""
Experiment description and details

We are trying to find the best representation for both topology and hyper-params and how to integrate them together and
feed them as input to our RNN model as a time-series data, we will use an a adjacencies matrix to encode the topology
and a long vector for all the layers HP`s in time step t and will input them both to an RNN based model at each time step
in order to predict the test-set accuracy of the model.

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

# def load_model_training_data(model_id):
#     model_df = pd.read_csv(data_folder + 'model-' + model_id + '/' + 'model-' + model_id + '.csv')
#     return model_df
#
#
# def fix_models_to_json(models):
#     for model_index, model in enumerate(models):
#         layers_list = ast.literal_eval(model)
#         for index, layer_str in enumerate(layers_list):
#             fixed_layer_str = fix_string_for_json(layer_str)
#             test_layer_to_json(fixed_layer_str)
#             layers_list[index] = fixed_layer_str
#             # print(layer_str)
#         # print(str(layers_list))
#         model = str(layers_list)
#         models[model_index] = model
#     return models
#
# def fix_string_for_json(_layer_str):
#     _layer_str = _layer_str.replace("\'", "\"")
#     _layer_str = _layer_str.replace("layer_type", "\"layer_type\"")
#     _layer_str = _layer_str.replace("rate", "\"rate\"")
#     _layer_str = _layer_str.replace("name", "\"name\"")
#     _layer_str = _layer_str.replace("axis", "\"axis\"")
#     _layer_str = _layer_str.replace("height", "\"height\"")
#     _layer_str = _layer_str.replace("width", "\"width\"")
#     _layer_str = _layer_str.replace("channels", "\"channels\"")
#     _layer_str = _layer_str.replace("stride", "\"stride\"")
#     _layer_str = _layer_str.replace("mode", "\"mode\"")
#     _layer_str = _layer_str.replace("momentum", "\"momentum\"")
#     _layer_str = _layer_str.replace("epsilon", "\"epsilon\"")
#     _layer_str = _layer_str.replace("activation_type", "\"activation_type\"")
#     _layer_str = _layer_str.replace("output_dim", "\"output_dim\"")
#     return _layer_str


# endregion
# region ================================ Load Model ===================================================================

all_models_df = pd.read_csv(data_folder + 'abstract_models.csv')
first_model = all_models_df.iloc[[0]]
model_id = first_model['model_id'][0]
model_topology_str = first_model['model_layers'][0]
model_training_data_df = utils.load_model_training_data(data_folder, model_id)

# endregion
# region ============================= Create Adj Matrix ===============================================================
# TODO - how should we address identity layers and their connections?


# endregion
# region ============================= Create HP vector ================================================================


# endregion
# region ============================= Create RNN model ================================================================

# endregion
print('dsfs')

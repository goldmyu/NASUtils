import copy
import json
import os
import uuid

import pandas as pd

from NNLayers import *


# ================================ UTILS ===============================================================================


def print_model_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure:
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
        print(attrs)
    print('-----------------------------------------------')


# ================================= Models Creation ====================================================================

def random_model(max_network_depth, attempets_num):
    layer_collection = []
    prev_layer = None
    for i in range(max_network_depth):
        prev_layer = random_layer(prev_layer)
        layer_collection.append(prev_layer)
    if check_legal_model(layer_collection):
        return layer_collection, attempets_num
    else:
        return random_model(max_network_depth, attempets_num + 1)


def random_layer(prev_layer):
    if prev_layer is None:
        # TODO - add support for linear layers
        # layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer, LinearLayer]
        layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
        return layers[random.randint(0, len(layers) - 1)]()

    # make sure no illegal modes are allowed like - dropout and dropout again etc
    else:
        if isinstance(prev_layer, DropoutLayer):
            layers = [BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
            return layers[random.randint(0, len(layers) - 1)]()

        elif isinstance(prev_layer, BatchNormLayer):
            layers = [DropoutLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
            return layers[random.randint(0, len(layers) - 1)]()

        elif isinstance(prev_layer ,ActivationLayer):
            layers = [DropoutLayer, BatchNormLayer, ConvLayer, PoolingLayer, IdentityLayer]
            return layers[random.randint(0, len(layers) - 1)]()

        elif isinstance(prev_layer ,PoolingLayer):
            layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, IdentityLayer]
            return layers[random.randint(0, len(layers) - 1)]()

    layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
    return layers[random.randint(0, len(layers) - 1)]()



# ========================== Model Validate and FIX ====================================================================

def fix_layers_dims(layer, prev_layer):
    # TODO - check layers dims with regard to previous layer and fix - Support Conv and max-pool and Linear
    print("ERRor - function not implemented")
    pass


def check_legal_model(layer_collection):
    # TODO - add checks to this method, doesnt validate network properly
    height = config['dataset_height']
    width = config['dataset_width']
    for layer in layer_collection:
        if type(layer) == ConvLayer or type(layer) == PoolingLayer:
            height = (height - layer.height) / layer.stride + 1
            width = (width - layer.width) / layer.stride + 1
        if height < 1 or width < 1:
            # print(f"illegal model, height={height}, width={width}")
            return False

    # check no illegal connections
    prev_layer = None
    for layer in layer_collection:
        if prev_layer is None:
            pass
        elif isinstance(layer, prev_layer):
            print('illegal model, prev layer is {} and current layer is {}'.format(prev_layer, layer))
            return False
        else:
            prev_layer = layer
    return True


# TODO -  This method is not used in this class, being called from NASUTILS, maybe we need to move it to NASUTILS
def finalize_model(layer_collection):
    if config['grid']:
        # TODO - add support for grid (parallel layers, skip connections)
        # return ModelFromGrid(layer_collection)
        pass
    else:
        # TODO - shouldnt we add input layer here ???
        layer_collection = copy.deepcopy(layer_collection)
        output_layer = LinearLayer(config['num_classes'])
        activation = ActivationLayer('softmax')
        layer_collection.append(output_layer)
        layer_collection.append(activation)
    return layer_collection


def get_model_true_depth(model):
    model_depth = 0
    if config['grid']:
        # TODO - add support for parallel layers and skip connections
        print('Error - add support for parallel layers and skip connections')
        pass
    else:
        for layer in model:
            if not isinstance(layer, IdentityLayer):
                model_depth += 1
    return model_depth


# ========================== Population Creation =======================================================================

# def initialize_population():
#     population = []
#     total_attempts = 0
#     print('initializing population...')
#
#     if config['grid']:
#         # TODO - support parallel layers and skip connections
#         # model_init = random_grid_model
#         pass
#     else:
#         pop_size = config['population_size']
#         for model_num in range(pop_size):
#             model_id = uuid.uuid4()
#             new_rand_model, attemptes_num = random_model(config['max_network_depth'], attempets_num=0)
#             population.append({'model_id': model_id, 'model': new_rand_model})
#             total_attempts += attemptes_num
#
#         print('Generated {} random models, avarege number of attempets per model creation was {}'.
#               format(pop_size, total_attempts / pop_size))
#
#     save_abstract_models_to_csv(population)
#     return population


def generate_abstract_model():
    model_id = uuid.uuid4()
    model, attemptes_num = random_model(config['max_network_depth'], attempets_num=0)
    model = finalize_model(model)
    print('Generated model {} number of attempts for creation was {}'.format(model_id, attemptes_num))
    return model_id, model


def save_configuration_as_json(models_save_path):
    conf_file = models_save_path + '/config.json'
    with open(conf_file, 'w') as fp:
        json.dump(config, fp)
        print('Configuration file was saved as JSON to {}'.format(conf_file))


def save_abstract_model_to_csv(model, model_id, model_test_accuracy, num_of_train_epochs):
    abstract_models_df = pd.DataFrame(columns=['model_id', 'model_layers',
                                               'model_depth', 'num_of_train_epochs',
                                               'model_test_accuracy'])

    abstract_models_df = abstract_models_df.append(
        {'model_id': model_id,
         'model_layers': [str(layer) for layer in model],
         'model_depth': get_model_true_depth(model),
         'num_of_train_epochs': num_of_train_epochs,
         'model_test_accuracy': model_test_accuracy,
         }, ignore_index=True)

    models_save_path = os.path.dirname(config['models_save_path'])
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)

    models_csv_file = models_save_path + '/abstract_models.csv'
    if os.path.isfile(models_csv_file):
        abstract_models_df.to_csv(models_csv_file, mode='a', header=False, index=False)
        print('Abstract Model {} was saved to {} file'.format(model_id, models_csv_file))

    else:
        abstract_models_df.to_csv(models_csv_file, index=False)
        print('CSV file was created with name {} and Model {} was added to it'.format(model_id,models_csv_file))
        save_configuration_as_json(models_save_path)

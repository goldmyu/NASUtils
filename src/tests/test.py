import ast
import json
import pandas as pd
import numpy as np


# ================================= Util Methods =======================================================================


def fix_models_to_json(models):
    for model_index, model in enumerate(models):
        layers_list = ast.literal_eval(model)
        for index, layer_str in enumerate(layers_list):
            fixed_layer_str = fix_string_for_json(layer_str)
            test_layer_to_json(fixed_layer_str)
            layers_list[index] = fixed_layer_str
            # print(layer_str)
        # print(str(layers_list))
        model = str(layers_list)
        models[model_index] = model
    return models


def fix_string_for_json(_layer_str):
    _layer_str = _layer_str.replace("\'", "\"")
    _layer_str = _layer_str.replace("layer_type", "\"layer_type\"")
    _layer_str = _layer_str.replace("rate", "\"rate\"")
    _layer_str = _layer_str.replace("name", "\"name\"")
    _layer_str = _layer_str.replace("axis", "\"axis\"")
    _layer_str = _layer_str.replace("height", "\"height\"")
    _layer_str = _layer_str.replace("width", "\"width\"")
    _layer_str = _layer_str.replace("channels", "\"channels\"")
    _layer_str = _layer_str.replace("stride", "\"stride\"")
    _layer_str = _layer_str.replace("mode", "\"mode\"")
    _layer_str = _layer_str.replace("momentum", "\"momentum\"")
    _layer_str = _layer_str.replace("epsilon", "\"epsilon\"")
    _layer_str = _layer_str.replace("activation_type", "\"activation_type\"")
    _layer_str = _layer_str.replace("output_dim", "\"output_dim\"")
    return _layer_str


def test_layer_to_json(_fixed_layer_str):
    try:
        json.loads(_fixed_layer_str)
    except Exception as e:
        print("This string {} cannot be converted to JSON\n Error was {}".format(_fixed_layer_str, e))
        raise e


def extract_models_to_csv(_models, _models_accuracy):
    models_layers_df = pd.DataFrame(columns=list_of_keys)

    for model_index, model in enumerate(_models):
        _dict = dict.fromkeys(list_of_keys)
        _dict['model_test_accuracy'] = _models_accuracy[model_index]

        layers_list = ast.literal_eval(model)
        for index, layer_str in enumerate(layers_list):
            json_layer = json.loads(layer_str)
            _dict['Layer_' + str(index + 1)] = json_layer.get('layer_type')
        models_layers_df = models_layers_df.append(_dict, ignore_index=True)
        print("Model #{} was added to CSV".format(model_index))
    models_layers_df.to_csv(file_path + experiment_name + '_extracted_data.csv', index=False)


def convert_categorical_to_num(_models_df):
    layer_types = ['PoolingLayer', 'DropoutLayer', 'ConvLayer', 'IdentityLayer',
                   'BatchNormLayer', 'LinearLayer', 'ActivationLayer']

    for col_name in _models_df.columns:
        if _models_df[col_name].dtype == 'object':
            _models_df[col_name] = _models_df[col_name].astype('category')
            _models_df[col_name] = _models_df[col_name].cat.codes
    _models_df.to_csv(file_path + experiment_name + '_extracted_data_numerical.csv', index=False)
    return _models_df


# ========================== Main Section ==============================================================================
list_of_keys = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6', 'Layer_7'
    , 'Layer_8', 'Layer_9', 'Layer_10', 'Layer_11', 'Layer_12', 'Layer_13', 'Layer_14'
    , 'Layer_15', 'Layer_16', 'Layer_17', 'Layer_18', 'Layer_19', 'Layer_20', 'Layer_21',
                'Layer_22', 'model_test_accuracy']

experiment_name = 'experiment_4_simple'
file_path = './generated_files/' + experiment_name + '/'
# models_df = pd.read_csv(file_path + '/abstract_models.csv')
models_df = pd.read_csv(file_path + experiment_name + '_extracted_data.csv')

# models = models_df['model_layers'].to_numpy()
models_accuracy = models_df['model_test_accuracy'].to_numpy()
# fixed_models = fix_models_to_json(models)
# extract_models_to_csv(fixed_models, models_accuracy)

# models_df_origin = models_df.copy(deep=True)
# models_df = convert_categorical_to_num(models_df)
























print('All models statistics : for {} models \nmax accuracy is {}'
      '\nmin accuracy is {}\nMean accuracy is {}\nVar of accuracy is {}'
      .format(len(models_accuracy), models_accuracy.max(),
              models_accuracy.min(), np.mean(models_accuracy), np.var(models_accuracy)))

epochs = models_df['num_of_train_epochs'].to_numpy()
print('\nEpochs statistics : \nMax num of epochs {}\nAvarage number of training epochs {}'.
      format(np.max(epochs), np.mean(epochs)))

model_depths = models_df['model_depth'].to_numpy()
print('\nModel depth statistics : \nMax num of layers {}\nAvarage number of layers {}'.
      format(np.max(model_depths), np.mean(model_depths)))


# model = load_pytorch_model(load_path='./generated_files/experiment_3_simple/'
#                                      'model-95bae87f-ea12-4fe1-8c8a-d7840d3e146b/'
#                                      'pytorch_model-95bae87f-ea12-4fe1-8c8a-d7840d3e146b.pt')
# mean var max min std which_dist{norm, log_norm, uniform, }
#
# wrong modes
# activ -> activ
# drop -> drop
# batch -> batch
# pool -> pool

import ast
import json
import pandas as pd

list_of_keys = ['Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6', 'Layer_7'
    , 'Layer_8', 'Layer_9', 'Layer_10', 'Layer_11', 'Layer_12', 'Layer_13', 'Layer_14'
    , 'Layer_15', 'Layer_16', 'Layer_17', 'Layer_18', 'Layer_19', 'Layer_20', 'Layer_21',
                'Layer_22']

layer_type_enumerate = {'IdentityLayer': 0, 'ConvLayer': 1, 'BatchNormLayer': 2, 'DropoutLayer': 3, 'PoolingLayer': 4,
                        'ActivationLayer': 5, 'LinearLayer': 6}


# region ============================= Util Methods ====================================================================

def load_model_training_data(data_folder, model_id):
    model_df = pd.read_csv(data_folder + 'model-' + model_id + '/' + 'model-' + model_id + '.csv')
    return model_df


def model_layers_str_to_dict(model_layers_str):
    layers_dict = dict.fromkeys(list_of_keys)
    layers_list = ast.literal_eval(model_layers_str)
    for index, layer_str in enumerate(layers_list):
        fixed_layer_str = test_and_fix_string_for_json(layer_str)
        layers_list[index] = fixed_layer_str
        json_layer = json.loads(fixed_layer_str)
        layers_dict['Layer_' + str(index + 1)] = json_layer
    return layers_dict


def test_and_fix_string_for_json(_layer_str):
    try:
        if test_layer_to_json(_layer_str):
            return _layer_str
    except Exception as e:
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
        return True
    except Exception as e:
        print("This string {} cannot be converted to JSON\n Error was {}".format(_fixed_layer_str, e))
        raise e

# endregion

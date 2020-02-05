import ast
import json
import pandas as pd


# region ============================= Util Methods ====================================================================

def load_model_training_data(data_folder, model_id):
    model_df = pd.read_csv(data_folder + 'model-' + model_id + '/' + 'model-' + model_id + '.csv')
    return model_df


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

# endregion

import pandas as pd
import utils
import numpy as np


# ================================ Config ==============================================================================

data_folder = '../../generated_files/experiment_6/'

# ============================= Util Methods ===========================================================================


def create_step_index():
    iteration = 0
    epoch = 0
    step = 0
    model_training_data_df['step'] = 0

    # Add step indexing to number each training step by following the iteration and epoch values
    for index, row in model_training_data_df.iterrows():
        if iteration == row.get("iteration") and epoch == row.get('epoch'):
            model_training_data_df.at[index, 'step'] = step
        else:
            iteration = row.get("iteration")
            epoch = row.get('epoch')
            step += 1
            model_training_data_df.at[index, 'step'] = step
    return step


def create_model_hp_dataframe(_num_of_training_steps):
    hp_vectors_df = pd.DataFrame()
    # Iterate over layer_dict and compare each layer name to the layer in the training data base.
    # If it is the same layer - insert hp into hp_vector, else skip 25 indexes leaving zeros.
    for step in range(_num_of_training_steps):
        # Create large vector for all HP and initialize to zeros - size 25x22=550 - 25 hp types and 22 layers
        hp_vector = np.zeros(550)

        for index, (layer_number, layer) in enumerate(model_layers_dict.items()):
            layer_type = layer.get("layer_type")
            sub = model_training_data_df.loc[model_training_data_df['step'] == step]
            sub_only_data = sub.drop(['layer_name', 'iteration', 'epoch','step'], axis=1)
            sub_index = 0

            if layer_type == "BatchNormLayer" or layer_type == "ConvLayer" or layer_type == "LinearLayer":
                hp_vector[index*25: index*25 + 25] = sub_only_data.iloc[sub_index]
        hp_vectors_df = hp_vectors_df.append(pd.Series(hp_vector), ignore_index=True)
    return hp_vectors_df


# ==================================== Main ============================================================================

all_models_df = pd.read_csv(data_folder + 'abstract_models.csv')

all_models_hp_df = pd.DataFrame()
# Iterate over all models and create Hyper-params df that contained all the models training steps HP values
for model in all_models_df.iterrows():
    # first_model = all_models_df.iloc[[0]]
    model_index = model[0]
    model = model[1]
    model_id = model['model_id']
    model_layers_str = model['model_layers']
    model_training_data_df = utils.load_model_training_data(data_folder, model_id)

    # Convert model string to json
    model_layers_dict = utils.model_layers_str_to_dict(model_layers_str)

    num_of_training_steps = create_step_index()
    model_hp_vectors_df = create_model_hp_dataframe(num_of_training_steps)
    print('cool')

import gem
from gem.utils import graph_util, plot_util
# import gem_embedding

import pandas as pd

all_models_df = pd.read_csv('../generated_files/experiment_6/abstract_models.csv')

model_structure = all_models_df.loc[all_models_df['model_id'] == '00d42e3f-77ac-493e-ae35-68f1fff517c3']

model_df = pd.read_csv('../generated_files/experiment_6/model-00d42e3f-77ac-493e-ae35-68f1fff517c3/'
                       'model-00d42e3f-77ac-493e-ae35-68f1fff517c3.csv')

sub_model_df = model_df.loc[model_df['epoch'] == 0]
sub_model_df = sub_model_df.loc[sub_model_df['iteration'] == 0]

for item in model_structure['model_layers'].iteritems():
    print(item)
print(model_df)

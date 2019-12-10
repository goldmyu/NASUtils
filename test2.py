import ast
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



# ================================= Util Methods =======================================================================

# ================================= Main Method ========================================================================


# load CSV file
experiment_name = 'experiment_4_simple'
file_path = './generated_files/' + experiment_name + '/'
models_df = pd.read_csv(file_path + 'experiment_4_simple_extracted_data_numerical.csv')

# remove layers 21,22
models_df = models_df.drop(['Layer_21', 'Layer_22'], axis=1)

# split df to train and test
train, test = train_test_split(models_df, test_size=0.2)
x_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values

x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values


gbt = GradientBoostingRegressor()
gbt.fit(x_train,y_train)

pred = gbt.predict(x_test)
print(mean_squared_error(y_test, pred))


test['pred']=pred
test.to_csv('generated_files/experiment_4_simple/test.csv', index=False)

print(test)


import ast
import json
import pandas as pd
import numpy as np
import pytorch_model_train




model = pytorch_model_train.PytorchModel.load_pytorch_model(load_path='../generated_files/experiment_6/'
                                                         'model-00d42e3f-77ac-493e-ae35-68f1fff517c3/'
                                                         'pytorch_model-00d42e3f-77ac-493e-ae35-68f1fff517c3.pt')


print(model)
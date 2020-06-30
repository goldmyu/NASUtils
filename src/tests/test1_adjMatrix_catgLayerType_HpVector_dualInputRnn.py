import pandas as pd
import utils
import numpy as np
import torch
import torch.utils.data as Data


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
#
# all_models_df = pd.read_csv(data_folder + 'abstract_models.csv')
# first_model = all_models_df.iloc[[0]]
# model_id = first_model['model_id'][0]
# model_layers_str = first_model['model_layers'][0]
# model_training_data_df = utils.load_model_training_data(data_folder, model_id)
#
# # Convert model string to json
# model_layers_dict = utils.model_layers_str_to_dict(model_layers_str)

# endregion
# region ============================= Create Adj Matrix ===============================================================
# TODO - how should we address identity layers and their connections?
adj_matrix = create_and_populate_adj_matrix(model_layers_dict)
adj_matrix_as_vector = adj_matrix.flatten()

# endregion
# region ============================= Create HP vector ================================================================

# iteration = 0
# epoch = 0
# step = 0
# model_training_data_df['step'] = 0
#
# # Add step indexing to number each training step by following the iteration and epoch values
# for index, row in model_training_data_df.iterrows():
#     if iteration == row.get("iteration") and epoch == row.get('epoch'):
#         # model_training_data_df.set_value(index, 'step', step)
#         model_training_data_df.at[index, 'step'] = step
#     else:
#         iteration = row.get("iteration")
#         epoch = row.get('epoch')
#         step += 1
#         # model_training_data_df.set_value(index, 'step', step)
#         model_training_data_df.at[index, 'step'] = step
#
# num_of_training_steps = step
#
#
# hp_vectors_df = pd.DataFrame()
# # Iterate over layer_dict and compare each layer name to the layer in the training data base.
# # If it is the same layer - insert hp into hp_vector, else skip 25 indexes leaving zeros.
# for step in range(num_of_training_steps):
#     # Create large vector for all HP and initialize to zeros - size 25x22=550 - 25 hp types and 22 layers
#     hp_vector = np.zeros(550)
#
#     for index, (layer_number, layer) in enumerate(model_layers_dict.items()):
#         layer_type = layer.get("layer_type")
#         sub = model_training_data_df.loc[model_training_data_df['step'] == step]
#         sub_only_data = sub.drop(['layer_name', 'iteration', 'epoch','step'], axis=1)
#         sub_index = 0
#
#         if layer_type == "BatchNormLayer" or layer_type == "ConvLayer" or layer_type == "LinearLayer":
#             hp_vector[index*25: index*25 + 25] = sub_only_data.iloc[sub_index]
#     hp_vectors_df = hp_vectors_df.append(pd.Series(hp_vector), ignore_index=True)

# endregion
# region ============================= Create LSTM model ===============================================================
""" 
RNN model with many-to-one arch, because we assume fix length of sequence data - using LSTM type cell

I tried 2 cases:
 * One with two inputs (dual-input RNN) it gets an adj matrix and HP vector with zeros for layers with no params
 * Second - single input RNN, adj matrix flattened to vector and concatenated to the HP vector


How to evaluate?
 difference in accuracy and AUC between test-set and rnn prediction - run experiment again and also create AUC
 
 How we Train the RNN - maybe train many-to-many, every step, try to predict the loss on the train batch and 
 in the final step try and predict the test-set accuracy/loss
"""

# ===================== Hyper-params =========================================

torch.manual_seed(1)

# GPU and CUDA support check
print('torch version {}\n'.format(torch.__version__))
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print("CUDA available ? {} CUDA device is:{}".format(cuda_available, device))
if cuda_available:
    print('Number of available CUDA devices {}'.format(torch.cuda.device_count()))

# LSTM Hyper-params
lstm_num_layers = 2
input_dim = 550  # number of features - 550 hyper-params per 1 time-step - which is 1 network snapshot during training
hidden_dim = 1  # number of features in the hidden states
sequence_length = 30  # input sequence length - 30 is the number of snapshot taken during the network training

# Generated data size
train_size = 10000
test_size = 100

# training Hyper-params
train_batch_size = 100
test_batch_size = 1
learning_rate = 1
num_of_epochs = 100


# ===================== Define the model =========================================


class LstmModel(torch.nn.Module):

    def __init__(self, _input_dim, _hidden_dim, _lstm_num_layers, _device):
        super(LstmModel, self).__init__()
        self.hidden_dim = _hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.device = _device
        self.lstm = torch.nn.LSTM(input_size=_input_dim, hidden_size=_hidden_dim, num_layers=_lstm_num_layers,
                                  batch_first=True)

    def forward(self, sequence):
        # h0 = torch.zeros(lstm_num_layers, batch_size, hidden_dim).to(device)
        # c0 = torch.zeros(lstm_num_layers, batch_size, hidden_dim).to(device)

        out, (hn, cn) = self.lstm(sequence)
        last_output = hn[-1]
        output = torch.sigmoid(last_output)
        return output


class DatasetLSTM(Data.Dataset):
    """
        Support class for the loading and batching of sequences of samples

        Args:
            dataset (Tensor): Tensor containing all the samples
            sequence_length (int): length of the analyzed sequence by the LSTM
            transforms (object torchvision.transform): Pytorch's transforms used to process the data
    """

    #  Constructor
    def __init__(self, dataset, _sequence_length, _transforms=None):
        self.dataset = dataset
        self.seq_length = _sequence_length
        self.transforms = _transforms

    # Override total dataset's length getter
    def __len__(self):
        return self.dataset.__len__()

    # Override single items' getter
    def __getitem__(self, idx):
        seq_list = self.dataset.iloc[idx]
        _data = torch.empty(sequence_length, 1)
        for _iter in range(sequence_length):
            _data[_iter, :] = torch.Tensor([seq_list[_iter]])
        label = torch.as_tensor(seq_list[-1])
        return _data, label


# ======================== Model train and test =====================================


def model_train(_lstm_model, _num_of_epochs, _data_loader, _optimizer):
    for epoch in range(_num_of_epochs):
        epoch_loss = 0
        for step, (data_batch, labels) in enumerate(_data_loader):
            # We need to clear them out before each instance
            _lstm_model.zero_grad()

            # for CUDA
            data_batch = data_batch.to(device)
            labels = labels.to(device)

            # Run our forward pass
            sequence_pred_label = _lstm_model(data_batch)

            # Compute the loss, gradients, and update the parameters
            loss = loss_function(sequence_pred_label.squeeze(), labels)
            epoch_loss += loss

            # if step % 50 == 0:
            #     print("Iteration {} loss is {}".format(step, loss.item()))

            loss.backward()
            _optimizer.step()

        if epoch_loss < 0.27:
            print("model training is done, switching to testing the model...")
            break
        print("Training Epoch number ", epoch, " the epoch loss is : ", epoch_loss.item())


def model_test(_lstm_model, _test_set):
    test_loss = 0
    for data, labels in _test_set:
        data = data.to(device)
        labels = labels.to(device)

        sequence_pred_label = _lstm_model(data)
        loss = loss_function(sequence_pred_label, labels)
        test_loss += loss
        print("Model_Test - input sequence was {} predicted next number is {}".
              format(data[0], sequence_pred_label[0].item()))

    avg_loss = test_loss / len(_test_set.dataset)
    print("Model test avarage loss was {}".format(avg_loss.item()))


# ======================== Main Section =====================================

# Adj matrix as vector is of size 484
# Training meta-data per iteration is of size
train_dataset = DatasetLSTM(dataset=train_set, _sequence_length=sequence_length, _transforms=None)
train_data_loader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=2)

test_dataset = DatasetLSTM(dataset=test_set, _sequence_length=sequence_length, _transforms=None)
test_data_loader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)


lstm_model = LstmModel(input_dim, hidden_dim, lstm_num_layers, device).to(device)
optimizer = torch.optim.Adadelta(params=lstm_model.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss().to(device)

print('End')
# endregion


import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as transforms

import pandas as pd
import random

torch.manual_seed(1)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print("CUDA available ? {}\nCUDA device is:{}".format(cuda_available, device))

# =========================================== Hyper-params =============================================================

# Generated data size
train_size = 10000
test_size = 100

# LSTM Hyper-params
lstm_num_layers = 5
input_dim = 1  # number of features
hidden_dim = 1  # number of features in the hidden states
sequence_length = 5  # input sequence length

# training Hyper-params
train_batch_size = 100
test_batch_size = 1
learning_rate = 1
num_of_epochs = 100

# =========================================== Define the model =========================================================


class LSTM_model(torch.nn.Module):

    def __init__(self, _input_dim, _hidden_dim, _lstm_num_layers, _device):
        super(LSTM_model, self).__init__()
        self.hidden_dim = _hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.device = _device
        self.lstm = torch.nn.LSTM(input_size=_input_dim, hidden_size=_hidden_dim, num_layers=_lstm_num_layers, batch_first=True)

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
    def __init__(self, dataset, sequence_length, transforms=None):
        self.dataset = dataset
        self.seq_length = sequence_length
        self.transforms = transforms

    # Override total dataset's length getter
    def __len__(self):
        return self.dataset.__len__()

    # Override single items' getter
    def __getitem__(self, idx):
        seq_list = self.dataset.iloc[idx]

        _data = torch.empty(sequence_length, 1).to(device)
        for _iter in range(sequence_length):
            _data[_iter, :] = torch.Tensor([seq_list[_iter]])
        # _data = _data.view(1, -1).to(device)
        # _data = torch.FloatTensor()

        label = torch.as_tensor(seq_list[-1]).to(device)
        return _data, label


# =========================================== Model train and test =====================================================


def model_train(_lstm_model, _num_of_epochs, _data_loader, _optimizer):
    for i in range(_num_of_epochs):
        epoch_loss = 0
        iteration=0
        for data_batch, labels in _data_loader:
            # We need to clear them out before each instance
            _lstm_model.zero_grad()

            # Run our forward pass
            sequence_pred_label = _lstm_model(data_batch)

            # Compute the loss, gradients, and update the parameters
            loss = loss_function(sequence_pred_label.squeeze(), labels)
            epoch_loss += loss

            # iteration += 1
            # if iteration % 50 == 0:
            #     print("Iteration {} loss is {}".format(iteration, loss.item()))

            loss.backward()
            _optimizer.step()

        if epoch_loss < 0.27:
            print("model training is done, switching to testing the model...")
            break
        print("Training Epoch number ", i, " the epoch loss is : ", epoch_loss.item())


def model_test(_lstm_model, _test_set):
    test_loss = 0
    for data, labels in _test_set:
        sequence_pred_label = _lstm_model(data)
        loss = loss_function(sequence_pred_label, labels)
        test_loss += loss
        print("Model_Test - input sequence was {} predicted next number is {}".
              format(data[0], sequence_pred_label[0].item()))

    avg_loss = test_loss / len(_test_set.dataset)
    print("Model test avarage loss was {}".format(avg_loss.item()))


# =========================================== Create the data -=========================================================

generated_data_sequence = [i / float(100) for i in range(100)]

# train data-set
train_set = pd.DataFrame(columns=['t1', 't2', 't3', 't4', 't5', 'sequence_label'])
for i in range(train_size):
    rand_index = random.randint(0, len(generated_data_sequence) - sequence_length - 1)
    temp_sequence = generated_data_sequence[rand_index: rand_index + sequence_length]
    temp_sequence_label = generated_data_sequence[rand_index + sequence_length]
    df_temp = pd.DataFrame({'t1': temp_sequence[0],
                            't2': temp_sequence[1],
                            't3': temp_sequence[2],
                            't4': temp_sequence[3],
                            't5': temp_sequence[4],
                            'sequence_label': temp_sequence_label}, index=[0])
    train_set = train_set.append(df_temp, ignore_index=True)

# test data-set
test_set = pd.DataFrame(columns=['t1', 't2', 't3', 't4', 't5', 'sequence_label'])
for i in range(test_size):
    rand_index = random.randint(0, len(generated_data_sequence) - sequence_length - 1)
    temp_sequence = generated_data_sequence[rand_index: rand_index + sequence_length]
    temp_sequence_label = generated_data_sequence[rand_index + sequence_length]
    df_temp = pd.DataFrame({'t1': temp_sequence[0],
                            't2': temp_sequence[1],
                            't3': temp_sequence[2],
                            't4': temp_sequence[3],
                            't5': temp_sequence[4],
                            'sequence_label': temp_sequence_label}, index=[0])
    test_set = test_set.append(df_temp, ignore_index=True)

# =========================================== Define the model =========================================================


# data_transform = transforms.Lambda(lambda x: listToTensor(x))
train_dataset = DatasetLSTM(dataset=train_set, sequence_length=sequence_length, transforms=None)
train_data_loader = Data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0)

test_dataset = DatasetLSTM(dataset=test_set, sequence_length=sequence_length, transforms=None)
test_data_loader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

lstm_model = LSTM_model(input_dim, hidden_dim, lstm_num_layers, device).to(device)
optimizer = torch.optim.Adadelta(params=lstm_model.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss().to(device)

model_train(lstm_model, num_of_epochs, train_data_loader, optimizer)
model_test(lstm_model, test_data_loader)

print('End')

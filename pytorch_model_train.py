import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
# from torchsummary import summary
# from tensorboardX import SummaryWriter
# from torch.utils.data import Dataset

from config import config
import logging

# ============================= General Settings =======================================================================

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

print('torch version {}\ntorchvision version {}'.format(torch.__version__, torchvision.__version__))
print('CUDA available? {} Device is {}'.format(cuda_available, device))

np.set_printoptions(threshold=sys.maxsize)
activations = {}

logger = logging.getLogger('pytorch')


# ========================== Helper methods ============================================================================

class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


def set_logger(model_id):
    logger.setLevel(logging.DEBUG)

    save_path = config['models_save_path'] + "/training_logs/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set StreamHandler to print to stdout INFO msgs
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setLevel(logging.INFO)
    stream.addFilter(InfoFilter())
    logger.addHandler(stream)

    # Other msgs will be logged to file (DEBUG)
    file_handler = logging.FileHandler(save_path + 'model-' + str(model_id) + '.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.propagate = False


# For TensorBoard Visualization
def write_model_summery(model, model_id, train_loader):
    writer = SummaryWriter('generated_files/visualization/' + str(model_id))
    # writer = SummaryWriter('generated_files/visualization/')
    # writer = SummaryWriter()

    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid)
    writer.add_graph(model=model, input_to_model=images, verbose=True)
    writer.flush()
    writer.close()


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def set_model_activation_output(model):
    for name, _layer in model._modules.items():
        _layer.register_forward_hook(get_activation(name))


# ========================== Main methods ============================================================================


def set_train_and_test_model(model, model_id):
    set_logger(model_id)

    # TODO - this is for testing purpose only - remove that after
    # model = Net()

    if cuda_available:
        model = model.cuda()
        logger.info('model parameters are cuda ? ' + str(next(model.parameters()).is_cuda))

    valid_size = config['validation_size']
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # TODO -  These values are dedicated for CIFAR10 dataset need to generalize
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010], )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data-sets/cifar10', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data-sets/cifar10', train=False,
                                           download=True, transform=transform)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed()
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                               sampler=train_sampler, num_workers=2, drop_last=True)

    validate_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], sampler=valid_sampler,
                                                  num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # Train and then test the model
    train_model(model, model_id, train_loader, validate_loader)
    test_model(model, model_id, test_loader)


def train_model(model, model_id, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss().cuda(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # TODO - put this call back when running the experiment
    # write_model_summery(model, model_id, train_loader)

    # Set hooks to get layers activations values
    set_model_activation_output(model)

    logger.info('Start training for model {}\nModel Summary\n{}'.format(model_id, model))
    logging_rate = config['logging_rate_initial']
    prev_valid_loss = float('inf')
    max_num_of_epochs = config['max_num_of_epochs']

    for epoch in range(max_num_of_epochs):  # loop over the dataset multiple times
        logging_rate = logging_rate * (epoch + 1)
        running_loss = 0.0
        epoch_correctly_labeled = 0
        total = 0

        logger.info('Started training epoch {}/{}\n'
                    'Logging rate every {} iterations'.format(epoch, max_num_of_epochs, logging_rate))

        for iter, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if cuda_available:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            batch_correctly_labeled = (predicted == labels).sum().item()
            epoch_correctly_labeled += batch_correctly_labeled

            # logging the network stats according to the logging rate
            if iter % logging_rate == 0:

                # General training statistics
                logger.info('Training stats ========= Epoch %d/%d ========= Iteration %d/%d ========= '
                            'Batch Accuracy %.3f ========= loss %.3f =========' % (
                                epoch + 1, max_num_of_epochs, iter, len(train_loader),
                                batch_correctly_labeled / config['batch_size'], running_loss / logging_rate))

                running_loss = 0.0

                # Layers weights, biases and gradients to log
                for layer_name, layer_params in model.named_parameters():
                    logger.debug('{layer_name : ' + layer_name + ',\nlayer_shape: ' + str(list(layer_params.size())) +
                                 ',\nvalues:' + str(layer_params.data.cpu().numpy()) + ',\ngradient_values:' +
                                 str(layer_params.grad.cpu().numpy()) + '}')

                # Layers activations values to log
                logger.debug('All layers activations values:\n')
                for layer_name, layer_activation in activations.items():
                    logger.debug('layer_name : ' + layer_name + '\nactivation_values: '
                                 + str(layer_activation.data.cpu().numpy()))

        logger.info('Epoch {} Accuracy is {}'.format(epoch + 1, epoch_correctly_labeled / total))

        stop_flag, prev_valid_loss = validation_check(epoch, model, model_id, prev_valid_loss, valid_loader)
        if stop_flag:
            break

    logger.info('Finished Training')


def validation_check(epoch, model, model_id, prev_valid_loss, valid_loader):
    _, curr_valid_loss = test_model(model=model, model_id=model_id,
                                    data_loader=valid_loader, validation_flag=True)

    if epoch > config['min_num_of_epochs']:
        if curr_valid_loss >= prev_valid_loss:
            logger.info('Early stopping criteria is meet, stopping training\n'
                        'current validation loss is {} previous validation loss {}'.format(curr_valid_loss,
                                                                                           prev_valid_loss))

            return True, prev_valid_loss
        else:
            return False, curr_valid_loss


def test_model(model, model_id, data_loader, validation_flag=False):
    criterion = nn.CrossEntropyLoss().cuda(device=device)
    correctly_labeled = 0
    total_predictions = 0
    running_loss = 0

    with torch.no_grad():
        for data in data_loader:

            images, labels = data
            if cuda_available:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correctly_labeled += (predicted == labels).sum().item()

    model_accuracy = correctly_labeled / total_predictions

    if validation_flag:
        logger.info('Model {} validation set accuracy is {}'.format(model_id, model_accuracy))
        return model_accuracy, running_loss
    else:
        logger.info('Model {} test set accuracy is {}'.format(model_id, model_accuracy))

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.batch_nrm1 = nn.BatchNorm2d(6)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.batch_nrm2 = nn.BatchNorm2d(16)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(self.batch_nrm1(F.relu(self.conv1(x))))
#         x = self.pool(self.batch_nrm2(F.relu(self.conv2(x))))
#
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

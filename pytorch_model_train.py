import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
# from tensorboardX import SummaryWriter


from torch.utils.data import Dataset
from config import config
import logging

# ============================= General Settings =======================================================================

print(torch.__version__)
print(torchvision.__version__)

np.set_printoptions(threshold=sys.maxsize)

save_path = "generated_files/training_logs/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = logging.getLogger('pytorch')
logger.setLevel(logging.DEBUG)


# stream = logging.StreamHandler()
# logger.addHandler(stream)


# ========================== Helper methods ============================================================================

def set_logger(model_id):
    file_handler = logging.FileHandler(save_path + 'model-' + str(model_id) + '.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


activations = {}


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

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data-sets/cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data-sets/cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    train_model(model, model_id, trainloader)
    test_model(model, testloader)


def train_model(model, model_id, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # ============================== TensorBoard Visualization ===============================================

    # writer = SummaryWriter('generated_files/visualization/' + str(model_id))
    writer = SummaryWriter('generated_files/visualization/')
    # writer = SummaryWriter()

    images, labels = next(iter(trainloader))
    grid = torchvision.utils.make_grid(images)

    writer.add_image('images', grid)
    writer.add_graph(model, images)
    writer.flush()
    writer.close()
    # os.system('tensorboard --logdir=generated_files/visualization/')

    # ========================================================================================================

    # Set hooks to get layers activations values
    set_model_activation_output(model)

    logger.info('Start training for model {}'.format(model_id))
    logger.info('Model Summary:\n' + str(model))
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for iterations, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # printing the network stats every 50 iterations
            if iterations % 50 == 0:
                # General training data
                logger.info(
                    'Training_stats - Epoch %d, Iteration %d, loss %.3f' % (
                    epoch + 1, iterations + 1, running_loss / 200))

                # Printing all layers weights and biases to log
                for layer_name, layer_params in model.named_parameters():
                    logger.info('{layer_name : ' + layer_name + ',\nlayer_shape: ' + str(list(layer_params.size())) +
                                ',\nvalues:' + str(layer_params.data.numpy()) + ',\ngradient_values:' +
                                str(layer_params.grad.numpy()) + '}')

                # Printing all layers activations values to log
                logger.info('All layers activations values:\n')
                for layer_name, layer_activation in activations.items():
                    logger.info(
                        'layer_name : ' + layer_name + '\nactivation_values: ' + str(layer_activation.data.numpy()))

                # print statistics
                # running_loss += loss.item()
                # if i % 200 == 199:  # print every 200 mini-batches
                #     print('[Epoch : %d Iteration : %5d loss: %.3f]' %
                #           (epoch + 1, i + 1, running_loss / 200))
                #     running_loss = 0.0

    print('Finished Training')


def test_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

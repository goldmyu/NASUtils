import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn

# from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import Dataset
from config import config
import logging

# ============================= Logger Settings ========================================================================


save_path = "generated_files/training_logs/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = logging.getLogger('pytorch')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ========================================================================================================


def set_logger(model_id):
    file_handler = logging.FileHandler(save_path + 'model-' + str(model_id) + '.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def train_model(model, model_id):
    set_logger(model_id)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data-sets/cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data-sets/cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    logger.info('Start training for model {}'.format(model_id))
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[Epoch : %d Iteration : %5d loss: %.3f]' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

            # printing the network weights every 50 iterations
            if i % 50 == 0:
                logger.info('[Epoch : %d Iteration : %5d loss: %.3f]' % (epoch + 1, i + 1, running_loss / 200))
                for layer_name, layer_params in model.named_parameters():
                    logger.info(msg=[layer_name, layer_params.size(), layer_params])

                for layer_name, layer_params in model.named_modules():
                    print(layer_params)
                    pass

    print('Finished Training')

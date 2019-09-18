import os
import sys
import time

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
import logging.handlers

# ============================= General Settings =======================================================================

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

print('torch version {}\ntorchvision version {}'.format(torch.__version__, torchvision.__version__))
print('CUDA available? {} Device is {}'.format(cuda_available, device))
if cuda_available:
    print('Number of available CUDA devices {}'.format(torch.cuda.device_count()))

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


class PytorchModel:

    def __init__(self, model, model_id, model_num):
        self.model = model
        self.model_id = model_id
        self.save_path = config['models_save_path'] + "model-" + str(model_id) + "/"
        self.activations = {}
        self.logger = logging.getLogger('pytorch_' + str(model_num))

    # ========================== Helper methods ============================================================================

    def set_logger(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # formatter = logging.Formatter('%(threadName)s %(message)s')

        # Set StreamHandler to print to stdout INFO msgs
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(logging.INFO)
        stream.addFilter(InfoFilter())
        # stream.setFormatter(formatter)

        # Other msgs will be logged to file (DEBUG)
        file_handler = logging.FileHandler(filename=self.save_path + 'model-' + str(self.model_id) + '.log')
        file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)

        # MemoryHandler gets all msgs saves them in memory and fluch when capacity is reached
        memory_handler = logging.handlers.MemoryHandler(capacity=1024 * 1000000000,
                                                        flushLevel=logging.DEBUG,
                                                        target=file_handler)

        # Queue handler and listner
        # log_queue = queue.Queue(-1)
        # queue_handler = logging.handlers.QueueHandler(log_queue)
        # listener = logging.handlers.QueueListener(log_queue, file_handler)
        # listener.start()

        # logger.addHandler(file_handler)
        # logger.addHandler(queue_handler)
        self.logger.addHandler(memory_handler)
        self.logger.addHandler(stream)

    # For TensorBoard Visualization
    def write_model_summery(self, train_loader):
        writer = SummaryWriter(self.save_path)

        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid)
        writer.add_graph(model=self.model.cpu(), input_to_model=images, verbose=True)
        writer.flush()
        writer.close()

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def set_model_activation_output(self):
        for name, _layer in self.model._modules.items():
            _layer.register_forward_hook(self.get_activation(name))

    def save_pytorch_model(self, optimizer, loss, epoch):
        # TODO - save pytorch model checkpoint
        save_file = self.save_path + 'pytorch_model-' + str(self.model_id) + '.pt'

        # torch.save(model, save_file)

        # torch.save({
        #     'model': model,
        #     'optimizer': optimizer,
        #     'epoch': epoch,
        #     'loss': loss
        # }, save_file)

        torch.save(obj={
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss, }
            , f=save_file)

    @staticmethod
    def load_pytorch_model(load_path):
        model = nn.Sequential()
        model = nn.DataParallel(model)

        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = optim.Adam(params=model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # - or -
        # model.train()
        return model

    # ========================== Main methods ============================================================================

    def set_train_and_test_model(self):
        self.set_logger()

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

        num_workers = config['num_of_dataloader_workers']

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                                   sampler=train_sampler, num_workers=num_workers, drop_last=True)

        validate_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], sampler=valid_sampler,
                                                      num_workers=num_workers)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], num_workers=num_workers)

        self.write_model_summery(train_loader)

        if cuda_available:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
            self.model.to(device)
            self.logger.info('model parameters are cuda ? ' + str(next(self.model.parameters()).is_cuda))

        # Train and then test the model
        num_of_train_epochs = self.train_model(train_loader, validate_loader)
        model_test_accuracy = self.test_model(test_loader)

        return model_test_accuracy, num_of_train_epochs

    def train_model(self, train_loader, valid_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Set hooks to get layers activations values
        self.set_model_activation_output()

        self.logger.info('Start training for model {}\nModel Summary\n{}'.format(self.model_id, self.model))
        logging_rate = config['logging_rate_initial']
        prev_valid_loss = float('inf')
        max_num_of_epochs = config['max_num_of_epochs']
        num_of_train_epochs = max_num_of_epochs

        for epoch in range(max_num_of_epochs):  # loop over the dataset multiple times
            logging_rate = logging_rate * (epoch + 1)
            running_loss = 0.0
            epoch_correctly_labeled = 0
            epoch_total_labeled = 0

            self.logger.info('\nStarted training epoch {}/{}\n'
                             'Logging rate every {} iterations'.format(epoch + 1,
                                                                       max_num_of_epochs,
                                                                       min(logging_rate, len(train_loader))))

            for _iter, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                if cuda_available:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                epoch_total_labeled += labels.size(0)
                batch_correctly_labeled = (predicted == labels).sum().item()
                epoch_correctly_labeled += batch_correctly_labeled

                # logging the network stats according to the logging rate
                if _iter % logging_rate == 0:
                    self.logger.info('Training stats ========= Epoch {}/{} ========= Iteration {}/{} ========= '
                                     'Batch Accuracy {} ========= loss {} ========='.
                                     format(epoch + 1, max_num_of_epochs, _iter, len(train_loader),
                                            batch_correctly_labeled / config['batch_size'],
                                            running_loss / logging_rate))

                    running_loss = 0.0

                    # Layers weights, biases and gradients to log
                    # print('Started logging weights|biases|gradients'.format())
                    # weight_strt = time.time()
                    # for layer_name, layer_params in self.model.named_parameters():
                    #     self.logger.debug('layer_name {},\nlayer_shape {}\nvalues {} \ngradient_values {}'.
                    #                  format(layer_name, list(layer_params.size()), layer_params.data, layer_params.grad))
                    # print('Finished logging weights|biases|gradients - time it took was {}Sec\n'.
                    #       format(round(time.time()-weight_strt)))

                    # TODO - put activation logging back in in the future
                    # Layers activations values to log
                    # logger.debug('All layers activations values:\n')
                    # print('Started logging activations'.format(time.time()))
                    # activ_strt = time.time()
                    # for layer_name, layer_activation in activations.items():
                    #     logger.debug('layer_name {}\nactivation_values {}'.
                    #                  format(layer_name, layer_activation.data))
                    # print('Finished logging activations - time it took was {}'.format(time.time()-activ_strt))

            self.logger.info('Epoch {} Accuracy is {}'.format(epoch + 1, epoch_correctly_labeled / epoch_total_labeled))

            stop_flag, prev_valid_loss = self.validate_model(epoch, prev_valid_loss, valid_loader)
            if stop_flag:
                num_of_train_epochs = epoch
                break

        self.logger.info('Finished Training')
        self.save_pytorch_model(optimizer, loss, num_of_train_epochs)
        return num_of_train_epochs

    def validate_model(self, epoch, prev_loss, valid_loader):
        accuracy, curr_loss = self.test_model(data_loader=valid_loader, validation_flag=True)
        self.logger.info('Validation set -- epoch {} -- accuracy {} -- loss {} -- previous loss {}'.
                         format(epoch + 1, accuracy, curr_loss, prev_loss))

        if epoch > config['min_num_of_epochs'] and curr_loss >= prev_loss:
            self.logger.info('Early stopping criteria is meet, stopping training after {} epochs'.format(epoch + 1))
            return True, prev_loss
        else:
            return False, curr_loss

    def test_model(self, data_loader, validation_flag=False):
        criterion = nn.CrossEntropyLoss().cuda()
        correctly_labeled = 0
        total_predictions = 0
        running_loss = 0

        with torch.no_grad():
            for data in data_loader:

                images, labels = data
                if cuda_available:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correctly_labeled += (predicted == labels).sum().item()

        model_accuracy = correctly_labeled / total_predictions

        if validation_flag:
            return model_accuracy, running_loss
        else:
            self.logger.info('Test set accuracy is {}'.format(model_accuracy))
            return model_accuracy

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

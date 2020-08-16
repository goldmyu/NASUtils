import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# from architecture_generator.config import config
import logging
import logging.handlers

import config

# region ============================= General Settings ================================================================
torch.multiprocessing.set_sharing_strategy('file_system')

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

print('torch version {}\ntorchvision version {}'.format(torch.__version__, torchvision.__version__))
print('CUDA available? {} Device is {}'.format(cuda_available, device))
if cuda_available:
    print('Number of available CUDA devices {}\n'.format(torch.cuda.device_count()))

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)


# endregion

class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


class PytorchModel:

    def __init__(self, model, model_id, model_num):
        self.model = model
        self.model_id = model_id
        self.save_path = config.models_save_path + "model-" + str(model_id) + "/"
        self.activations = {}
        self.logger = logging.getLogger('pytorch_' + str(model_num))
        self.training_info_df = pd.DataFrame(columns=['layer_name',
                                                      'weight_max', 'weight_min', 'weight_mean', 'weight_var',
                                                      'weight_std',
                                                      'bias_max', 'bias_min', 'bias_mean', 'bias_var', 'bias_std',
                                                      'weight_gradient_max', 'weight_gradient_min',
                                                      'weight_gradient_mean', 'weight_gradient_var',
                                                      'weight_gradient_std',
                                                      'bias_gradient_max', 'bias_gradient_min', 'bias_gradient_mean',
                                                      'bias_gradient_var', 'bias_gradient_std',
                                                      'activation_max', 'activation_min', 'activation_mean',
                                                      'activation_var', 'activation_std',
                                                      'iteration', 'epoch'])

    # region ========================== Helper methods =================================================================

    def set_logger(self):
        dirname = os.path.dirname(__file__)
        self.save_path = dirname + "/../.." + self.save_path

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

        # MemoryHandler gets all msgs saves them in memory and flush when capacity is reached
        memory_handler = logging.handlers.MemoryHandler(capacity=1024 * 1000000000,
                                                        flushLevel=logging.DEBUG,
                                                        target=file_handler)

        # Queue handler and listner
        # log_queue = queue.Queue(-1)
        # queue_handler = logging.handlers.QueueHandler(log_queue)
        # listener = logging.handlers.QueueListener(log_queue, file_handler)
        # listener.start()

        # logger.addHandler(file_handler)
        # self.logger.addHandler(queue_handler)
        self.logger.addHandler(memory_handler)
        self.logger.addHandler(stream)

    def write_model_summery(self, train_loader):
        writer = SummaryWriter(self.save_path)
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid)
        writer.add_graph(model=self.model.cpu(), input_to_model=images, verbose=False)
        writer.close()

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    def set_model_activation_output(self):
        if isinstance(self.model, nn.DataParallel):
            for name, _layer in self.model.module._modules.items():
                _layer.register_forward_hook(self.get_activation(name))
        else:
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

    @staticmethod
    def log_weights_biases_gradients(_temp_dict, layer_name_and_type, layer_params):
        layer_data = layer_params.data
        _temp_dict[layer_name_and_type[1] + '_max'] = torch.max(layer_data).item()
        _temp_dict[layer_name_and_type[1] + '_min'] = torch.min(layer_data).item()
        _temp_dict[layer_name_and_type[1] + '_mean'] = torch.mean(layer_data).item()
        _temp_dict[layer_name_and_type[1] + '_var'] = torch.var(layer_data).item()
        _temp_dict[layer_name_and_type[1] + '_std'] = torch.std(layer_data).item()

        layer_grad = layer_params.grad
        _temp_dict[layer_name_and_type[1] + '_gradient_max'] = torch.max(layer_grad).item()
        _temp_dict[layer_name_and_type[1] + '_gradient_min'] = torch.min(layer_grad).item()
        _temp_dict[layer_name_and_type[1] + '_gradient_mean'] = torch.mean(layer_grad).item()
        _temp_dict[layer_name_and_type[1] + '_gradient_var'] = torch.var(layer_grad).item()
        _temp_dict[layer_name_and_type[1] + '_gradient_std'] = torch.std(layer_grad).item()

        return _temp_dict

    def log_activations(self, _temp_dict):
        for layer_name_act, layer_activation in self.activations.items():
            if layer_name_act == _temp_dict['layer_name']:
                # TODO - check 'goodness of fit' test against Unifrom, Normal, logNormal distributions
                _temp_dict['activation_max'] = torch.max(layer_activation).item()
                _temp_dict['activation_min'] = torch.min(layer_activation).item()
                _temp_dict['activation_mean'] = torch.mean(layer_activation).item()
                _temp_dict['activation_var'] = torch.var(layer_activation).item()
                _temp_dict['activation_std'] = torch.std(layer_activation).item()
                break
        return _temp_dict

    def log_training_info(self, epoch, max_num_of_epochs, _iter,
                          iter_per_epoch, batch_correctly_labeled, logging_rate,
                          running_loss, train_epoch_accu, train_epoch_loss,
                          val_loss, val_accu,
                          at_epoch_end=False):

        temp_dict = {}
        # This happens for every layer in the model
        for layer_name, layer_params in self.model.named_parameters():
            layer_name_and_type = layer_name.split('.')

            if layer_name_and_type[0] == 'module':
                layer_name_and_type = layer_name_and_type[1:]

            if 'layer_name' in temp_dict.keys():
                # This is for bias
                if config.log_weights:
                    temp_dict = self.log_weights_biases_gradients(temp_dict, layer_name_and_type, layer_params)
                    self.training_info_df = self.training_info_df.append(temp_dict, ignore_index=True)
                    temp_dict = {}
            else:
                # this is for weights
                temp_dict['layer_name'] = layer_name_and_type[0]
                temp_dict['epoch'] = epoch
                temp_dict['iteration'] = _iter
                temp_dict['val_accu'] = val_accu
                temp_dict['val_loss'] = val_loss
                temp_dict['train_epoch_accu'] = train_epoch_accu
                temp_dict['train_epoch_loss'] = train_epoch_loss
                if at_epoch_end:
                    temp_dict['at_epoch_end'] = 1
                else:
                    temp_dict['at_epoch_end'] = 0

                if config.log_weights:
                    temp_dict = self.log_weights_biases_gradients(temp_dict, layer_name_and_type, layer_params)
                if config.log_activations:
                    temp_dict = self.log_activations(temp_dict)

    # endregion

    # region ========================== Main methods ===================================================================

    def set_train_and_test_model(self):
        self.set_logger()

        valid_size = config.validation_size
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

        dir_name = os.path.dirname(__file__)
        trainset = torchvision.datasets.CIFAR10(root=dir_name + '/../../data-sets/cifar10', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root=dir_name + '/../../data-sets/cifar10', train=False,
                                               download=True, transform=transform)

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        np.random.seed()
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        num_workers = config.num_of_dataloader_workers

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                                   sampler=train_sampler, num_workers=num_workers, drop_last=True)

        validate_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, sampler=valid_sampler,
                                                      num_workers=num_workers)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, num_workers=num_workers)

        self.write_model_summery(train_loader)

        if cuda_available:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
            self.model.to(device)
            self.logger.debug('model parameters are cuda ? ' + str(next(self.model.parameters()).is_cuda))

        # Train and then test the model
        num_of_train_epochs = self.train_model(train_loader, validate_loader)
        model_test_accuracy = self.test_model(test_loader)

        return model_test_accuracy, num_of_train_epochs

    def train_model(self, train_loader, valid_loader):
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(self.model.parameters())

        # Set hooks to get layers activations values
        self.set_model_activation_output()

        self.logger.debug('Start training for model {}\nModel Summary\n{}'.format(self.model_id, self.model))
        logging_rate = config.logging_rate_initial
        prev_valid_loss = float('inf')
        max_num_of_epochs = config.max_num_of_epochs
        num_of_train_epochs = max_num_of_epochs

        for epoch in range(max_num_of_epochs):  # loop over the dataset multiple times
            logging_rate = logging_rate * (epoch + 1)
            running_loss = 0.0
            train_epoch_loss = 0
            epoch_correctly_labeled = 0
            epoch_total_labeled = 0
            iter_per_epoch = 0
            batch_correctly_labeled = 0

            self.logger.info('\nStarted training epoch {}/{}'.format(epoch + 1, max_num_of_epochs))

            if not config.log_only_at_epoch_end:
                self.logger.info('Logging rate every {} iterations'.format(min(logging_rate, len(train_loader))))

            for _iter, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                iter_per_epoch = len(train_loader)
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
                train_epoch_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                epoch_total_labeled += labels.size(0)
                batch_correctly_labeled = (predicted == labels).sum().item()
                epoch_correctly_labeled += batch_correctly_labeled

                # TODO - resume this once logging every few iterations instead of logging at the end of epoch alone
                # if _iter % logging_rate == 0 and config.log_only_at_epoch_end:
                #     self.logger.info('Training stats ========= Epoch {}/{} ========= Iteration {}/{} ========= '
                #                          'Batch Accuracy {} ========= loss {} ========='.
                #                          format(epoch + 1, max_num_of_epochs, _iter, iter_per_epoch,
                #                                 batch_correctly_labeled / config.batch_size,
                #                                 running_loss / logging_rate))
                #
                #     self.log_training_info(
                #         epoch=epoch, max_num_of_epochs=max_num_of_epochs,
                #         _iter=-1, iter_per_epoch=iter_per_epoch,
                #         batch_correctly_labeled=batch_correctly_labeled,
                #         logging_rate=logging_rate, running_loss=running_loss,
                #         val_loss=0, val_accu=0,
                #         at_epoch_end=False
                #     )
                #     running_loss = 0.0

            # end of an epoch
            train_epoch_accu = epoch_correctly_labeled / epoch_total_labeled
            self.logger.info('Training Epoch {}/{} stats ========= Epoch Accuracy {} ========= Epoch loss {} ========='
                             .format(epoch + 1, max_num_of_epochs, train_epoch_accu, train_epoch_loss))

            # validation phase
            stop_flag, prev_valid_loss, val_accu = self.validate_model(epoch, max_num_of_epochs,
                                                                       prev_valid_loss, valid_loader)

            if config.log_only_at_epoch_end:
                self.log_training_info(epoch=epoch, max_num_of_epochs=max_num_of_epochs,
                                       _iter=-1, iter_per_epoch=-1,
                                       batch_correctly_labeled=-1,
                                       logging_rate=logging_rate, running_loss=-1,
                                       train_epoch_accu=train_epoch_accu, train_epoch_loss=train_epoch_loss,
                                       val_loss=prev_valid_loss, val_accu=val_accu,
                                       at_epoch_end=True)

            # TODO - change early stop to only after 5 epochs with no improvment instead of after 1
            if stop_flag:
                num_of_train_epochs = epoch
                break

        self.logger.info('Finished Training')
        self.training_info_df.to_csv(self.save_path + 'model-' + str(self.model_id) + '.csv', index=False)
        self.save_pytorch_model(optimizer, loss, num_of_train_epochs)
        return num_of_train_epochs

    def validate_model(self, epoch, max_num_of_epochs, prev_loss, valid_loader):
        val_accu, curr_loss = self.test_model(data_loader=valid_loader, validation_flag=True)
        self.logger.info('Validation set -- epoch {}/{} -- accuracy {} -- loss {} -- previous loss {}'.
                         format(epoch + 1, max_num_of_epochs, val_accu, curr_loss, prev_loss))

        if epoch > config.min_num_of_epochs and curr_loss >= prev_loss:
            self.logger.info('Early stopping criteria is meet, stopping training after {} epochs'.format(epoch + 1))
            return True, curr_loss, val_accu
        else:
            return False, curr_loss, val_accu

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
    # endregion

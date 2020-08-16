import torch
from torch import nn
import numpy as np
from NNLayers import *
import config


# ================================== Pytorch Code Genration ============================================================


def create_pytorch_model(layer_collection, model_id, apply_fix=False):
    model = nn.Sequential()
    activations_types = {'relu': nn.ReLU, 'elu': nn.ELU, 'softmax': nn.Softmax, 'sigmoid': nn.Sigmoid}
    input_shape = (config.batch_size, config.dataset_channels, config.dataset_height, config.dataset_width)

    for i in range(len(layer_collection)):
        layer = layer_collection[i]
        if i > 0:
            out = model.forward(input=torch.ones(size=input_shape, dtype=torch.float32))
            prev_layer_shape = out.cpu().data.numpy().shape
        else:
            prev_layer_shape = input_shape

        if isinstance(layer, PoolingLayer):
            if apply_fix:
                # TODO maybe we should fix the layers on the abstract representation?!?!
                fix_layers_dims(layer, prev_layer_shape)
            model.add_module(name=f'{type(layer).__name__}_{i}',
                             module=nn.MaxPool2d(kernel_size=(int(layer.height), int(layer.width)),
                                                 stride=int(layer.stride)))

        elif isinstance(layer, ConvLayer):
            if apply_fix:
                fix_layers_dims(layer, prev_layer_shape)
            model.add_module(f'{type(layer).__name__}_{i}', nn.Conv2d(in_channels=prev_layer_shape[1],
                                                                      out_channels=layer.channels,
                                                                      kernel_size=(layer.height, layer.width),
                                                                      stride=layer.stride))

        elif isinstance(layer, BatchNormLayer):
            model.add_module(f'{type(layer).__name__}_{i}', nn.BatchNorm2d(prev_layer_shape[1],
                                                                           affine=True, eps=1e-5))

        elif isinstance(layer, ActivationLayer):
            model.add_module(name=f'{type(layer).__name__}_{i}', module=activations_types[layer.activation_type]())

        elif isinstance(layer, DropoutLayer):
            model.add_module(f'{type(layer).__name__}_{i}', nn.Dropout(p=layer.rate))

        elif isinstance(layer, IdentityLayer):
            # why add a layer here? just skip it
            # model.add_module(f'{type(layer).__name__}_{i}', IdentityModule())
            # print('IdentityLayer - not adding layer to model')
            pass

        elif isinstance(layer, SqueezeLayer):
            model.add_module('squeeze', Squeeze_Layer())

        elif isinstance(layer, LinearLayer):
            model.add_module(f'Flatten_Layer_{i}', Flatten_Layer())
            model.add_module(f'{type(layer).__name__}_{i}', nn.Linear(in_features=np.prod(prev_layer_shape[1:])
                                                                      , out_features=layer.output_dim))

    # TODO - refactor weights init to another method - add support for choosing wether to init weights or not
    # init.xavier_uniform_(list(model._modules.items())[-3][1].weight, gain=1)
    # init.constant_(list(model._modules.items())[-3][1].bias, 0)
    print('Created pytorch model for model {}'.format(model_id))
    return model


def fix_layers_dims(layer, prev_layer):
    # TODO - Add this for grid support - check layers dims with regard to previous layer - Support Conv and max-pool
    pass


# ==================================== Pytorch Custom Layers ===========================================================

class Squeeze_Layer(nn.Module):
    def __init__(self):
        super(Squeeze_Layer, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class Flatten_Layer(nn.Module):
    def __init__(self):
        super(Flatten_Layer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs

import random
from config import config


# TODO - in some layers we have name and in some we dont, why is that?!?!?
class Layer():
    def __init__(self, name=None):
        self.name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__


class InputLayer(Layer):
    def __init__(self, shape_height, shape_width, name=None):
        Layer.__init__(self, name)
        self.shape_height = shape_height
        self.shape_width = shape_width

    def __str__(self):
        return f'{{layer_type: \'InputLayer\', shape_height:\'{self.shape_height}\', ' \
            f'shape_width: \'{self.shape_width}\',name:\'{self.name}\'}}'


# TODO - what does this layer do?
class SqueezeLayer(Layer):
    def __init__(self, name=None):
        Layer.__init__(self, name)

    def __str__(self):
        return f'{{layer_type: \'SqueezeLayer\',name:\'{self.name}\'}}'


class DropoutLayer(Layer):
    def __init__(self, rate=None, name=None):
        Layer.__init__(self, name)
        if rate is None:
            rate = random.uniform(0, config['dropout_max_rate'])
        self.rate = rate

    def __str__(self):
        return f'{{layer_type: \'DropoutLayer\',rate:\'{self.rate}\', name:\'{self.name}\'}}'


class BatchNormLayer(Layer):
    def __init__(self, axis=3, momentum=0.1, epsilon=1e-5, name=None):
        Layer.__init__(self, name)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

    def __str__(self):
        return f'{{layer_type: \'BatchNormLayer\', axis:\'{self.axis}\', ' \
            f'momentum: \'{self.momentum}\', epsilon: \'{self.epsilon}\',name:\'{self.name}\'}}'


class ActivationLayer(Layer):
    def __init__(self, activation_type='elu', name=None):
        Layer.__init__(self, name)
        self.activation_type = activation_type

    def __str__(self):
        return f'{{layer_type: \'ActivationLayer\', activation_type:\'{self.activation_type}\', name:\'{self.name}\'}}'


class LinearLayer(Layer):
    # @initializer
    def __init__(self, output_dim=None, name=None):
        Layer.__init__(self, name)
        if output_dim is None:
            output_dim = random.randint(1, config['linear_max_dim'])
        self.output_dim = output_dim

    def __str__(self):
        return f'{{layer_type: \'LinearLayer\', output_dim:\'{self.output_dim}\', name:\'{self.name}\'}}'

# TODO - do we add support for 'dilation' and 'padding' for a conv layer?
class ConvLayer(Layer):
    # @initializer
    def __init__(self, height=None, width=None, channels=None, stride=None, name=None):
        Layer.__init__(self, name)
        if height is None:
            height = random.randint(1, config['conv_max_height'])
        if width is None:
            width = random.randint(1, config['conv_max_width'])
        if channels is None:
            channels = random.randint(1, config['conv_max_channels'])
        if stride is None:
            stride = random.randint(1, config['conv_max_stride'])
        self.height = height
        self.width = width
        self.channels = channels
        self.stride = stride

    def __str__(self):
        return f'{{layer_type: \'ConvLayer\', height:\'{self.height}\', width:\'{self.width}\', ' \
            f'channels:\'{self.channels}\',stride:\'{self.stride}\', name:\'{self.name}\'}}'


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, height=None, width=None, stride=None, mode='max', name=None):
        Layer.__init__(self, name)
        if height is None:
            height = random.randint(1, config['pool_max_height'])
        if width is None:
            width = random.randint(1, config['pool_max_width'])
        if stride is None:
            stride = random.randint(1, config['pool_max_stride'])
        self.height = height
        self.width = width
        self.stride = stride
        self.mode = mode

    def __str__(self):
        return f'{{layer_type: \'PoolingLayer\', height:\'{self.height}\', width:\'{self.width}\', ' \
            f'stride:\'{self.stride}\', mode:\'{self.mode}\' , name:\'{self.name}\'}}'


class IdentityLayer(Layer):
    def __init__(self, name=None):
        Layer.__init__(self, name)

    def __str__(self):
        return f'{{layer_type: \'IdentityLayer\',name:\'{self.name}\'}}'


class ZeroPadLayer(Layer):
    def __init__(self, height_pad_top, height_pad_bottom, width_pad_left, width_pad_right, name=None):
        Layer.__init__(self, name)
        self.height_pad_top = height_pad_top
        self.height_pad_bottom = height_pad_bottom
        self.width_pad_left = width_pad_left
        self.width_pad_right = width_pad_right

    def __str__(self):
        return f'{{layer_type: \'ZeroPadLayer\', height_pad_top:\'{self.height_pad_top}\', ' \
            f'height_pad_bottom:\'{self.height_pad_bottom}\', width_pad_left:\'{self.width_pad_left}\', ' \
            f'width_pad_right:\'{self.width_pad_right}\', name:\'{self.name}\'}}'


class ConcatLayer(Layer):
    def __init__(self, first_layer_index, second_layer_index, name=None):
        Layer.__init__(self, name)
        self.first_layer_index = first_layer_index
        self.second_layer_index = second_layer_index

    def __str__(self):
        return f'{{layer_type: \'ConcatLayer\', first_layer_index:\'{self.first_layer_index}\', ' \
            f'second_layer_index:\'{self.second_layer_index}\', name:\'{self.name}\'}}'


# TODO - what does this layer do?!?!
# class AveragingLayer(Layer):
#     def __init__(self):
#         pass

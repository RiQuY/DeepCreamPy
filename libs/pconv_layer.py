
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, InputSpec
import tensorflow as tf

class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError(
                'The channel dimension of the inputs should be defined.'
            )

        self.input_dim = input_shape[0][channel_axis]

        kernel_shape = self.kernel_size + (self.input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='img_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        # IMPORTANT:
        # Fixed tensor, NOT a Keras weight.
        self.kernel_mask = tf.ones(
            shape=kernel_shape,
            dtype=self.dtype
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, mask=None):

        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PartialConvolution2D must be called on a list of two tensors '
                '[img, mask]. Instead got: ' + str(inputs)
            )

        # Original normalization
        normalization = K.mean(inputs[1], axis=[1, 2], keepdims=True)
        normalization = K.repeat_elements(
            normalization, inputs[1].shape[1], axis=1
        )
        normalization = K.repeat_elements(
            normalization, inputs[1].shape[2], axis=2
        )

        # Image convolution
        img_output = K.conv2d(
            (inputs[0] * inputs[1]) / normalization,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Mask convolution
        mask_output = K.conv2d(
            inputs[1],
            self.kernel_mask,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        mask_output = K.cast(
            K.greater(mask_output, 0),
            'float32'
        )

        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import backend
from tensorflow.python.keras.layers.advanced_activations import PReLU


def relu(x):
    return layers.ReLU()(x)

# default value is 'zero'
# 0.25 may lead to less effective learning, see: https://arxiv.org/pdf/1502.01852.pdf
def prelu(x):
    return PReLU(alpha_initializer=tf.initializers.constant(0.25))(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# squeeze excitation module
def _se_block(inputs, filters, se_ratio, prefix):
    # compress and restore -> [!!!] one different
    # 类似 bottleneck，提升 model capacity，减少参数和运算量r^2
    x = layers.GlobalAveragePooling2D(name=prefix + 'se/avgp')(
        inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
        # print("[Se] channel first")
    else:
        x = layers.Reshape((1, 1, filters))(x)
        # print("[Se] channel last")

    # squeeze stage
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'sr/cn')(
        x)
    x = layers.ReLU(name=prefix + 'se/Relu')(x)
    # excitation stage
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'se/cn1')(
        x)
    x = hard_sigmoid(x)
    # final scale -> reweighting the feature maps
    x = layers.Multiply(name=prefix + 'se/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, block_id, activation, se_ratio=None):
    # backup input tensor -> res
    shortcut = x

    prefix = block_id + 'ne_cn/'
    # get input channels' num -> decide if expand
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    infilters = backend.int_shape(x)[channel_axis]
    midfilters = _depth(infilters * expansion)
    print(
        f"[INRES] {block_id}: channel_axis: {channel_axis}, in_filters: {infilters}, mid_filters: {midfilters}, out_filter: {filters}")

    # Point-wise expansion
    if midfilters > infilters:
        # Expand
        prefix = block_id + 'ex_cn/'
        x = layers.Conv2D(
            midfilters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'ex')(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'ex/bn')(x)
        x = activation(x)

    # Depth-wise convolution
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'dw')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'dw/bn')(x)
    x = activation(x)

    if se_ratio is not None and se_ratio > 0:
        x = _se_block(x, midfilters, se_ratio, prefix)

    # Projection convolution
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'pj')(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'pj/bn')(x)

    # Residual part
    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def smaller_conv_7x7(x, filters, expansion, activation, se_ratio, name):
    block_out_filters = filters // 3

    # three 3x3 convolution -> similar with one 7x7 convolution
    x1 = _inverted_res_block(x, expansion, block_out_filters, 3, 1, name + "_ir1/", activation, se_ratio)
    x2 = _inverted_res_block(x1, expansion, block_out_filters, 3, 1, name + "_ir2/", activation, se_ratio)
    x3 = _inverted_res_block(x2, expansion, block_out_filters, 3, 1, name + "_ir3/", activation, se_ratio)
    x = layers.concatenate([x1, x2, x3], 3)

    return x


def create_OpenposeSginglent(Debug=False):
    # create basic model
    backbone_model = MobileNetV3Large(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False
    )

    # get block5 and block 11
    layers_names = [
        "expanded_conv_5/Add",
        "expanded_conv_11/Add"
    ]
    inter_layers = [backbone_model.get_layer(name).output for name in layers_names]
    backend_layer = keras.Model(inputs=backbone_model.inputs, outputs=inter_layers)
    backend_layer.trainable = False

    if Debug:
        print("backend layer's structure is as followed:")
        print(backend_layer.summary())

    # create whole model input layer
    inputs = keras.Input(shape=(224, 224, 3))

    # get output of backbone model
    back_inter5, back_inter11 = backend_layer(inputs, training=True)
    if Debug:
        print(f"[SHAPE] inter 5:{back_inter5.shape}; inter 11:{back_inter11.shape}")
    back_inter11 = tf.image.resize(back_inter11, (back_inter11.shape[1] * 2, back_inter11.shape[2] * 2))
    if Debug:
        print(f"[SHAPE] inter 5:{back_inter5.shape}; inter 11:{back_inter11.shape}")
    back_out = tf.concat([back_inter5, back_inter11], 3)

    # s 0 -- paf 0
    x = smaller_conv_7x7(back_out, 192, 1.0, prelu, None, "s0_cn0")
    x = smaller_conv_7x7(x, 192, 1.0, prelu, None, "s0_cn1")
    x = smaller_conv_7x7(x, 192, 1.0, prelu, None, "s0_cn2")
    x = smaller_conv_7x7(x, 192, 1.0, prelu, None, "s0_cn3")
    x = smaller_conv_7x7(x, 192, 1.0, prelu, None, "s0_cn4")
    x = _inverted_res_block(x, 1.0, 256, 1, 1, "s0_ir0/", prelu, 0.25)
    s0_out = layers.Conv2D(38, 1, 1)(x)

    # s 1 -- paf 1
    x = layers.concatenate([back_out, s0_out], 3)
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s1_cn0")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s1_cn1")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s1_cn2")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s1_cn3")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s1_cn4")
    x = _inverted_res_block(x, 1.0, 512, 1, 1, "s1_ir0/", prelu, 0.25)
    s1_out = layers.Conv2D(38, 1, 1)(x)

    # s 2 -- paf 2
    x = layers.concatenate([back_out, s1_out], 3)
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s2_cn0")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s2_cn1")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s2_cn2")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s2_cn3")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s2_cn4")
    x = _inverted_res_block(x, 1.0, 512, 1, 1, "s2_ir0/", prelu, 0.25)
    s2_out = layers.Conv2D(38, 1, 1)(x)

    # s 3 -- heatmap
    x = layers.concatenate([back_out, s2_out], 3)
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s3_cn0")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s3_cn1")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s3_cn2")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s3_cn3")
    x = smaller_conv_7x7(x, 384, 1.0, prelu, None, "s3_cn4")
    x = _inverted_res_block(x, 1.0, 512, 1, 1, "s3_ir0/", prelu, 0.25)
    s3_out = layers.Conv2D(19, 1, 1)(x)

    # compress all layers into one model
    model = keras.Model(
        inputs=inputs,
        outputs=[s0_out, s1_out, s2_out, s3_out],
        name="singlenet"
    )

    # model = keras.Model(inputs=inputs,outputs=x,name="test")
    print("single model's structure as followed:")
    print(model.summary())

    return model, backend_layer


if __name__ == '__main__':
    model, backend_layer = create_OpenposeSginglent(True)
    print(f"trainable: {backend_layer.trainable}")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer


class MOperation:
    def __init__(self, input_tenors_names):
        self.input_tenors_names = input_tenors_names

    def get_input_tenors_names(self):
        return self.input_tenors_names

    def get_out_chs_number(self, input_chs_list):
        raise NotImplemented()

    def __call__(self, inputs):
        raise NotImplemented()


class UpscaleX2(MOperation):

    def __init__(self, input_tenors_names):
        super(UpscaleX2, self).__init__(input_tenors_names)

    def get_out_chs_number(self, input_chs_list):
        return input_chs_list[0]

    def __call__(self, inputs):
        b, w, h, c = inputs.shape
        return tf.image.resize(inputs, (w * 2, h * 2))


class MobileLayer(Layer):

    def __init__(self, name, mobile_v3_model):
        super().__init__(name=name)
        print(f"MobileNet Layer starts to init")

        mobile_v3_model.layers.pop()
        print(f"Pop the top layer, check the structure as followed:")
        self.backend_model = mobile_v3_model

        self.input_layer = self.backend_model.input
        self.x = self.backend_model.input

    def build(self, input_image):
        layer_names = [
            "expanded_conv_5/Add",
            "expanded_conv_11/Add"
        ]

        layers = [self.backend_model.get_layer(name).output for name in layer_names]

        self.backend_layer = Model(inputs=self.backend_model.inputs, outputs=layers)
        # self.backend_layer.trainable=False
        self.upscaleX2 = UpscaleX2("back-upScale")
        # self.b_concat = ConcatOper("back-outConcat")

        # print(f"check structure of backend layer:")
        # print(self.backend_layer.summary())

        # for layer in self.backend_layer.layers:
        #     layer.trainable=False
        #     print(f"layer {layer.name} has been set untrainable")


    def call(self, input_image, training=True):
        b5, b11 = self.backend_layer(input_image)
        b11_x2 = self.upscaleX2(b11)
        # b_out = self.b_concat([b5, b11_x2])
        b_out = tf.concat([b5, b11_x2], 3)

        return b_out

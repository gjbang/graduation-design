import os
from model_struct.singlenet_func import create_OpenposeSginglent
from tools.plot import probe_model_singlenet
import importlib
import tensorflow as tf



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def single_convert(converter,output_path,allow_cusops=False,optimizations=[],supported_ops=[]):
    converter.allow_cusops=allow_cusops
    converter.optimizations = optimizations
    converter.target_spec.supported_ops=supported_ops
    try:
        tflite_model = converter.convert()
    except BaseException:
        print("[INFO]", output_path, " fails to convert")
    else:
        open(output_path,"wb").write(tflite_model)
        print("[INFO]", output_path, " has been converted")


def export_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # WARNING:absl:Found untraced functions. These functions will not be directly callable after loading.
    tflite_file = output_path + "-nothing.tflite"
    single_convert(converter,tflite_file)

    # WARNING:absl:Found untraced functions. These functions will not be directly callable after loading.
    # dynamic range quantization
    optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_file = output_path + "-dynamic.tflite"
    single_convert(converter=converter,output_path=tflite_file,optimizations=optimizations)


    # dynamic range quantization
    # tf.float16 set the float only 16bits
    optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_file = output_path + "-custom_ops.tflite"
    single_convert(converter=converter,output_path=tflite_file,allow_cusops=True,optimizations=optimizations)


    # dynamic range quantization
    optimizations = [tf.lite.Optimize.DEFAULT]
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_file = output_path + "-opsSet_tf&in.tflite"
    single_convert(converter=converter, output_path=tflite_file, allow_cusops=True, optimizations=optimizations,supported_ops=supported_ops)


    # dynamic range quantization
    optimizations = [tf.lite.Optimize.DEFAULT]
    supported_ops = [tf.float16,tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_file = output_path + "-opsSet_tf&in-float16.tflite"
    single_convert(converter=converter, output_path=tflite_file, allow_cusops=True, optimizations=optimizations,supported_ops=supported_ops)


    # '''
    # [FAILS] exception
    # '''
    # # dynamic range quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # tf.float16 set the float only 16bits
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.allow_custom_ops = True
    #
    # try:
    #     tflite_file = output_path + "-opsSet_in-16w8.tflite"
    #     tflite_model = converter.convert()
    # except BaseException:
    #     print("[INFO]", tflite_file, " fails to convert")
    # else:
    #     open(tflite_file, "wb").write(tflite_model)
    #     print("[INFO]", tflite_file, " has been finished")


def main():

    tflite_path = "../resources/tflite_model"

    # register_tf_netbuilder_extensions()

    # load saved model

    # module = importlib.import_module('models')
    # create_model = getattr(module, create_model_fn)
    # model = create_model(pretrained=False)
    model,backend_layer = create_OpenposeSginglent()
    model.compile()
    root_dir = os.getcwd()
    print("[INFO] current root dir: ", root_dir)
    # weights_dir = os.path.join(root_dir, "model/")
    weights_dir ="../model/"
    print("[INFO] current ckpt dir: ", weights_dir)
    latest = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest)
    print("[INFO] load model finished")

    # first pass
    probe_model_singlenet(model, test_img_path="../resources/ski_224.jpg")
    print("[INFO] first pass has finished!")

    # export model to tflite
    export_to_tflite(model, tflite_path)
    print("[INFO] Done !!!")


if __name__ == '__main__':

    print(tf.__version__)
    main()
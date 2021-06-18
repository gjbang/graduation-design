import os
import tensorflow as tf
from dataset.dataflows import get_dataflow
from dataset.tfrecord_gen import create_tfrecord, read_tfrecord


# ds = tf.data.Dataset.from_generator(
#     gen(df),
#     output_signature=(
#         # the one is there to clarify that there is one of these objects
#         tf.TensorSpec(shape=(224,224,3), dtype=tf.float32),
#         tf.TensorSpec(shape=(28,28,38), dtype=tf.float32),
#         tf.TensorSpec(shape=(28,28,19), dtype=tf.float32),
#         tf.TensorSpec(shape=(224,224,3), dtype=tf.float32),
#         tf.TensorSpec(shape=(28,28,38), dtype=tf.float32),
#         tf.TensorSpec(shape=(28,28,19), dtype=tf.float32)
#     )
# )


def get_dataset(annot_path, img_dir, batch_size, lmdb_file, strict=False, x_size=224, y_size=28, use_o=False):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
                # yield tuple(j)

        return f

    # not use tfrecord
    # if lmdb_file is "None":
    df, size = get_dataflow(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=x_size,
        y_size=y_size,
        lmdb_file=lmdb_file,
        use_o = use_o
    )
    df.reset_state()

    print("[DEBUG] start to execute generator")
    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([x_size, x_size, 3]),
            tf.TensorShape([y_size, y_size, 38]),
            tf.TensorShape([y_size, y_size, 19])
        )
    )

        # ds = tf.data.Dataset.from_generator(
        #     gen(df), (tf.float32, tf.float32, tf.float32),
        #     output_shapes=(
        #         tf.TensorShape([2,x_size, x_size, 3]),
        #         tf.TensorShape([2,y_size, y_size, 38]),
        #         tf.TensorShape([2,y_size, y_size, 19])
        #     )
        # )
    # else:
    #
    #     if not os.path.exists(lmdb_file + "-0.tfrecord"):
    #         print(f"[DEBUG] {lmdb_file}-0.tfrecord not exists")
    #         # -- 是一个用于df单个数据读取的迭代器
    #         # --通过yield关键字将普通函数转换为了一个类似list的迭代器，对于海量数据可以有效节约内存
    #         # -- yield：每次执行到yield执行，函数即yield后的变量的值，执行结束，但同时保存函数此时的执行状态
    #         # -- 下次调用会从上次调用时的yield的下一条语句开始执行
    #         # -- 即 gen 实际上在返回df的单条数据集 的内容
    #
    #         df, size = get_dataflow(
    #             annot_path=annot_path,
    #             img_dir=img_dir,
    #             strict=strict,
    #             x_size=x_size,
    #             y_size=y_size,
    #             lmdb_file=lmdb_file
    #         )
    #         df.reset_state()
    #
    #         print("[DEBUG] start to execute generator")
    #
    #         # 第一个 tensor 是输入的图片 [image_size, image_size ,channel]
    #         # 第二个 tensor 是根据数据集置信得到的真实 paf 值
    #         # 第三个 tensor 是...............的真实 heatmap 值
    #         # 所以 gen(df) 的返回值应该是如下形式:
    #         # [ tensor[224,224,3], tensor[28,28,38], tensor[28,28,19] ]
    #         ds = tf.data.Dataset.from_generator(
    #             gen(df), (tf.float32, tf.float32, tf.float32),
    #             output_shapes=(
    #                 tf.TensorShape([2,x_size, x_size, 3]),
    #                 tf.TensorShape([2,y_size, y_size, 38]),
    #                 tf.TensorShape([2,y_size, y_size, 19])
    #             )
    #         )
    #         ds = ds.shuffle(batch_size * 2)
    #         print("[DEBUG] start to create tfrecord file")
    #         create_tfrecord(ds, lmdb_file)
    #
    #     ds = read_tfrecord(lmdb_file)
    #     if strict:
    #         size = 4396
    #     else:
    #         size = 106088
    # ds = ds.map(lambda x0, x1, x2,x3,x4,x5: ((x0, x1, x2),(x3,x4,x5)), num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.unbatch()
    # ds = ds.shuffle(batch_size)
    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.cache()
    # ds = ds.shuffle(batch_size * 2)
    # with tf.device("/gpu:0"):
    ds = ds.apply(tf.data.experimental.prefetch_to_device('/gpu:0', buffer_size=tf.data.AUTOTUNE))
    # ds = ds.prefetch(2)

    # 最终的ds引入第四个维度[0]=batch_size，表示单个tensor内有十条有效的数据
    return ds, size


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # annot_path_train = 'd://coco-dataset/annotations/person_keypoints_train2017.json'
    # img_dir_train = 'd:///coco-dataset/train2017/'
    annot_path_val = 'd:///coco-dataset/annotations/person_keypoints_val2017.json'
    img_dir_val = 'd://coco-dataset/val2017/'
    # annot_path_val = '/mnt/coco-dataset/annotations/person_keypoints_val2017.json'
    # img_dir_val = '/mnt/coco-dataset/val2017/'
    batch_size = 5

    # ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size)
    ds_val, ds_val_size = get_dataset(annot_path_val, img_dir_val, batch_size, strict=True, lmdb_file="None",use_o=True)
    cnt=0
    for i,x,y in ds_val:
    #     # print(len(i))
        print(i.shape)
        print(x.shape)
        print(y.shape)
        cnt +=1
        if cnt % 100 ==0:
            print(cnt)
    #     # break
    # print(cnt)
    # print(len(ds_val))

import os
import cv2
import numpy as np
import functools

import tensorflow as tf

from tensorpack.dataflow import MultiProcessMapDataZMQ, TestDataSpeed, MultiProcessMapData, MultiProcessRunner,MultiProcessRunnerZMQ,MultiProcessMapAndBatchDataZMQ
from tensorpack.dataflow.common import MapData
from tensorpack.dataflow import DataFlow,LMDBSerializer,TFRecordSerializer

# from dataset.augmentors import CropAug, FlipAug, ScaleAug, RotateAug, ResizeAug
# from dataset.base_dataflow import CocoDataFlow, JointsLoader
# from dataset.dataflow_steps import create_all_mask, augment, read_img, apply_mask, gen_mask
# from dataset.label_maps import create_heatmap, create_paf

from dataset.augmentors import CropAug, FlipAug, ScaleAug, RotateAug, ResizeAug
from dataset.base_dataflow import CocoDataFlow, JointsLoader
from dataset.dataflow_steps import create_all_mask, augment, read_img, apply_mask, gen_mask
from dataset.label_maps import create_heatmap, create_paf


class DivideDataFlow(DataFlow):
    def __init__(self,ds,y_size):
        self.ds=ds
        self.y_size=y_size

    def reset_state(self):
        self.ds.reset_state()

    def __iter__(self):
        for dps in self.ds:
            for dp in dps:
                img = dp[0]
                hp = create_heatmap(JointsLoader.num_joints_and_bkg, self.y_size, self.y_size,
                                    dp[1], 5.0, stride=8)

                paf = create_paf(JointsLoader.num_connections, self.y_size, self.y_size,
                                 dp[1], 0.8, stride=8)

                yield [img, paf, hp]


def build_sample(components, y_size):
    """
    Builds a sample for a model.

    :param components: components
    :return: list of final components of a sample.
    """
    # img = components[0][0]
    # aug_joints = components[0][1]

    img = components[0]
    aug_joints = components[1]

    # print(aug_joints)

    #
    heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
                             aug_joints, 5.0, stride=8)


    pafmap = create_paf(JointsLoader.num_connections, y_size, y_size,
                        aug_joints, 0.8, stride=8)

    #
    # o_img= components[1][0]
    # o_joints = components[1][1]
    # o_heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
    #                          o_joints, 5.0, stride=8)
    #
    # o_pafmap = create_paf(JointsLoader.num_connections, y_size, y_size,
    #                     o_joints, 0.8, stride=8)
    #
    # r_img=np.concatenate((np.expand_dims(img,axis=0),np.expand_dims(o_img,axis=0)),axis=0)
    # r_paf = np.concatenate((np.expand_dims(pafmap, axis=0), np.expand_dims(o_pafmap, axis=0)), axis=0)
    # r_hp = np.concatenate((np.expand_dims(heatmap, axis=0), np.expand_dims(o_heatmap, axis=0)), axis=0)
    # return [r_img,r_paf,r_hp]

    # return [img,o_img,pafmap,o_pafmap,heatmap,o_heatmap]

    # return [[img,pafmap,heatmap],
    # [o_img,o_pafmap,o_heatmap]]

    #
    # for dp in components:
    #     img = dp[0]
    #     hp = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
    #                              dp[1], 5.0, stride=8)
    #
    #     paf = create_paf(JointsLoader.num_connections, y_size, y_size,
    #                         dp[1], 0.8, stride=8)
    #
    #     yield [img,paf,hp]
    return [img,pafmap,heatmap]


def get_dataflow(annot_path, img_dir, strict, lmdb_file, x_size=224, y_size=28,use_o=False):
    """
    This function initializes the tensorpack dataflow and serves generator
    for training operation.

    :param annot_path: path to the annotation file
    :param img_dir: path to the images
    :return: dataflow object
    """
    coco_crop_size = 368

    # configure augmentors

    augmentors = [
        ScaleAug(scale_min=0.5,
                 scale_max=1.1,
                 target_dist=0.6,
                 interp=cv2.INTER_CUBIC),

        RotateAug(rotate_max_deg=40,
                  interp=cv2.INTER_CUBIC,
                  border=cv2.BORDER_CONSTANT,
                  border_value=(128, 128, 128), mask_border_val=1),

        CropAug(coco_crop_size, coco_crop_size, center_perterb_max=40, border_value=128,
                mask_border_val=1),

        FlipAug(num_parts=18, prob=0.5),

        ResizeAug(x_size, x_size)

    ]

    # prepare augment function
    # functools.partial() ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    # ??????????????????????????? augment() ????????????????????? augmentors ???????????????????????? augmentors
    # ??????????????????????????????????????????????????? augmentors ??????????????????????????????????????????????????????????????????
    augment_func = functools.partial(augment,
                                     augmentors=augmentors,
                                     use_o=use_o)

    # prepare building sample function

    build_sample_func = functools.partial(build_sample,
                                          y_size=y_size)

    # build the dataflow
    # -- in fact, dataflow is a class containing some vars, these vars usually are list or tuple
    # -- these lists or tuples contains single data belong one image in order of indices
    df = CocoDataFlow((coco_crop_size, coco_crop_size), annot_path, img_dir)
    df.prepare()
    size = df.size()
    print(f"[DEBUG] has prepared {size} coco-data meta")

    # ??? df ????????????????????????????????? df
    # ????????????????????? func????????????????????????????????????
    ## -- ????????????
    df = MapData(df, read_img)
    # df = MultiProcessMapDataZMQ(df, num_proc=2,map_func=read_img, strict=strict)
    print("[DEBUG] read img ok")
    # -- seems the Mapdata-df will convert into list-components when passed as params
    # print("test df == components, img_path == df[0]: ",str(df[0]))
    # print("[DEBUG] df's type: ",str(type(df)))
    df = MapData(df, augment_func)
    # df = MultiProcessMapDataZMQ(df, num_proc=4, map_func=augment_func, buffer_size=2000, strict=strict)
    print("[DEBUG] augment ok")
    # -- windows cannot use multiprocessMapData
    # -- because windows10 doesn't support zero MQ
    # -- although there is MultiProcessMapData, it's just the alias of MulitProcessMapDataZMQ
    df = MapData(df, build_sample_func)
    # df = DivideDataFlow(df,y_size=y_size)
    # df = MultiProcessRunner(df,num_proc=2,num_prefetch=1000)
    # df = MultiProcessRunnerZMQ(df,num_proc=12)
    # df = MultiProcessMapDataZMQ(df, num_proc=12, map_func=build_sample_func, buffer_size=size, strict=strict)
    # df = MultiProcessMapAndBatchDataZMQ(df, num_proc=12, map_func=build_sample_func, buffer_size=int(size//10), batch_size=batch_size)

    print("[DEBUG] build sample ok")
    # after inputing into MapData(), df's type become MapData -- a kind of class used for tensorflow
    # MapData -- ???????????????ds??????????????????func??????????????????????????????????????????ds??????
    # ????????????????????????????????????????????????????????????????????????MapData
    # print("[DEBUG] check joints in df: ",str(df.ds.ds.ds.all_meta[0].all_joints))

    # ?????????joints??????????????????18???joints???????????????

    # print(f"create lmdb_file: {lmdb_file}")
    # tfSerializer=TFRecordSerializer()
    # tfSerializer.save(df,lmdb_file)

    # return df, df.size()
    return df,size


if __name__ == '__main__':
    """
    Run this script to check speed of generating samples. Tweak the nr_proc
    parameter of PrefetchDataZMQ. Ideally it should reflect the number of cores 
    in your hardware
    """
    batch_size = 10
    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, 'd:/coco-dataset/annotations/person_keypoints_val2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, 'd:/coco-dataset/val2017/'))

    df1, size1 = get_dataflow(annot_path, img_dir, False, lmdb_file=None,  x_size=224, y_size=28)
    # df2, size2 = get_dataflow_vgg(annot_path, img_dir, False, x_size=368, y_size=46, include_outputs_masks=True)

    TestDataSpeed(df1, size=1000).start()
    # TestDataSpeed(df2, size=100).start()

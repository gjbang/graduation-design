import tensorflow as tf
import datetime
from datetime import timedelta
from timeit import default_timer as timer
from estimation.config import get_default_configuration

# base_dir = "D://coco-dataset"
# annot_path_train = base_dir + "/annotations/person_keypoints_train2017.json"
# annot_path_val = base_dir + "/annotations/person_keypoints_val2017.json"
# img_dir_train = base_dir + "/train2017/train2017/"
# img_dir_val = base_dir + "/val2017/val2017"

base_dir = "/root/coco-dataset"
annot_path_train = base_dir + "/annotations/person_keypoints_train2017.json"
annot_path_val = base_dir + "/annotations/person_keypoints_val2017.json"
img_dir_train = base_dir + "/train2017/"
img_dir_val = base_dir + "/val2017"

# model save path
checkpoints_folder = "./model/ckpt/"
output_weights = './model/weights/singlenet'
output_model = './model/structs/struct_model'

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_loss_heatmap = tf.keras.metrics.Mean('train_loss_heatmap', dtype=tf.float32)
train_loss_paf0 = tf.keras.metrics.Mean('train_loss_paf_0', dtype=tf.float32)
train_loss_paf1 = tf.keras.metrics.Mean('train_loss_paf_1', dtype=tf.float32)
train_loss_paf2 = tf.keras.metrics.Mean('train_loss_paf_2', dtype=tf.float32)

val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
val_loss_heatmap = tf.keras.metrics.Mean('val_loss_heatmap', dtype=tf.float32)
val_loss_paf0 = tf.keras.metrics.Mean('val_loss_paf_0', dtype=tf.float32)
val_loss_paf1 = tf.keras.metrics.Mean('val_loss_paf_1', dtype=tf.float32)
val_loss_paf2 = tf.keras.metrics.Mean('val_loss_paf_2', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs_singlenet/gradient_tape/train/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_log_dir = 'logs_singlenet/gradient_tape/train/val'
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

t_paf0_log = 'logs_singlenet/gradient_tape/train/paf0'
t_paf1_log = 'logs_singlenet/gradient_tape/train/paf1'
t_paf2_log = 'logs_singlenet/gradient_tape/train/paf2'
t_paf0_sw = tf.summary.create_file_writer(t_paf0_log)
t_paf1_sw = tf.summary.create_file_writer(t_paf1_log)
t_paf2_sw = tf.summary.create_file_writer(t_paf2_log)
v_paf0_log = 'logs_singlenet/gradient_tape/val/paf0'
v_paf1_log = 'logs_singlenet/gradient_tape/val/paf1'
v_paf2_log = 'logs_singlenet/gradient_tape/val/paf2'
v_paf0_sw = tf.summary.create_file_writer(v_paf0_log)
v_paf1_sw = tf.summary.create_file_writer(v_paf1_log)
v_paf2_sw = tf.summary.create_file_writer(v_paf2_log)

graph_log_dir = "logs_singlenet/func/record"
graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)

cfg=get_default_configuration()
plot_update_steps = 120


batch_size = 80
lr = 3e-4
train_epoch = 100
fn_epoch = 5

output_paf_idx = 2
output_heatmap_idx = 3


# @profiler
def update_scalar(epoch, step_per_epoch, cur_step):
    summary_step = (epoch - 1) * step_per_epoch + cur_step - 1
    with train_summary_writer.as_default():
        with tf.name_scope('tloss_a'):
            tf.summary.scalar('tloss_a', train_loss.result(), step=summary_step)
            tf.summary.scalar('tloss_a_heatmap', train_loss_heatmap.result(), step=summary_step)
        with tf.name_scope('tloss_paf'):
            tf.summary.scalar('tloss_paf_stage_2', train_loss_paf2.result(), step=summary_step)
            tf.summary.scalar('tloss_paf_stage_0', train_loss_paf0.result(), step=summary_step)
            tf.summary.scalar('tloss_paf_stage_1', train_loss_paf1.result(), step=summary_step)
    with t_paf0_sw.as_default():
        with tf.name_scope('tloss_paf'):
            tf.summary.scalar('tloss_paf_all', train_loss_paf0.result(), step=summary_step)
    with t_paf1_sw.as_default():
        with tf.name_scope('tloss_paf'):
            tf.summary.scalar('tloss_paf_all', train_loss_paf1.result(), step=summary_step)
    with t_paf2_sw.as_default():
        with tf.name_scope('tloss_paf'):
            tf.summary.scalar('tloss_paf_all', train_loss_paf2.result(), step=summary_step)


def update_val_scalar(epoch):
    val_loss_res = val_loss.result()
    val_loss_heatmap_res = val_loss_heatmap.result()
    val_loss_paf_res2 = val_loss_paf2.result()
    val_loss_paf_res1 = val_loss_paf1.result()
    val_loss_paf_res0 = val_loss_paf0.result()
    print(f'Validation losses for epoch: {epoch} : Loss paf {val_loss_paf_res2}, Loss heatmap '
          f'{val_loss_heatmap_res}, Total loss {val_loss_res}')

    with val_summary_writer.as_default():
        with tf.name_scope("val_loss_a"):
            tf.summary.scalar('val_loss_a', val_loss_res, step=epoch)
            tf.summary.scalar('val_loss_a_heatmap', val_loss_heatmap_res, step=epoch)
        with tf.name_scope("val_loss_paf"):
            tf.summary.scalar('val_loss_paf2', val_loss_paf_res2, step=epoch)
            tf.summary.scalar('val_loss_paf1', val_loss_paf_res1, step=epoch)
            tf.summary.scalar('val_loss_paf0', val_loss_paf_res0, step=epoch)
    with v_paf0_sw.as_default():
        with tf.name_scope('val_loss_paf'):
            tf.summary.scalar('val_loss_paf_all', val_loss_paf_res0, step=epoch)
    with v_paf1_sw.as_default():
        with tf.name_scope('val_loss_paf'):
            tf.summary.scalar('val_loss_paf_all', val_loss_paf_res1, step=epoch)
    with v_paf2_sw.as_default():
        with tf.name_scope('val_loss_paf'):
            tf.summary.scalar('val_loss_paf_all', val_loss_paf_res2, step=epoch)
    val_loss.reset_states()
    val_loss_heatmap.reset_states()
    val_loss_paf2.reset_states()
    val_loss_paf0.reset_states()
    val_loss_paf1.reset_states()
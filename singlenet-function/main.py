import os
import gc
# from memory_profiler import profile
import datetime
from datetime import timedelta
from timeit import default_timer as timer

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.eager import profiler

from train_config import *
from model_struct.singlenet_func import create_OpenposeSginglent

from dataset.generators import get_dataset
from tools.plot import plot_to_image, probe_model_singlenet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)

# @profiler
def update_train_loss(total_loss, losses):
    train_loss(total_loss)
    train_loss_heatmap(losses[3])
    train_loss_paf0(losses[0])
    train_loss_paf1(losses[1])
    train_loss_paf2(losses[2])


# @profiler
def reset_train_loss():
    train_loss.reset_states()
    train_loss_heatmap.reset_states()
    train_loss_paf0.reset_states()
    train_loss_paf1.reset_states()
    train_loss_paf2.reset_states()


# @profiler
def update_plot(model, cur_step):
    figure = probe_model_singlenet(model, test_img_path="resources/ski_224.jpg")
    with train_summary_writer.as_default():
        tf.summary.image("Test prediction", plot_to_image(figure), step=cur_step)


# @profiler
def save_ckpt(epoch, step, ckpt, manager):
    ckpt.step.assign(tf.cast(step, tf.int32))
    ckpt.epoch.assign(tf.cast(epoch, tf.int32))
    save_path = manager.save()
    tf.print("Saved checkpoint for step {}: {}".format(step, save_path))


# L2 loss function
# squared difference: x:[2,3] y:[4,6] -> z:[4,9]
def eucl_loss(y_true, y_pred):
    return tf.reduce_sum(tf.math.squared_difference(y_pred, y_true)) / batch_size / 2


@tf.function
def val_one_step(model, x, y_val_true0, y_val_true1):
    y_val_pred = model(x)
    losses = [eucl_loss(y_val_true0, y_val_pred[0]),
              eucl_loss(y_val_true0, y_val_pred[1]),
              eucl_loss(y_val_true0, y_val_pred[2]),
              eucl_loss(y_val_true1, y_val_pred[3])]
    total_loss = tf.reduce_sum(losses)

    return losses, total_loss


# @profiler
def val_one_epoch(ds_val, model, epoch):
    # calculate validation loss
    print("Calculating validation losses...")
    val_step = 0
    # o_it = iter(ods_val)
    start_t = timer()
    for x_val, y_val_true0, y_val_true1 in ds_val:
        val_step += 1


        if val_step % 100 == 0:
            print(f"Validation step {val_step} ...")
        losses, total_loss = val_one_step(model, x_val, y_val_true0, y_val_true1)
        val_loss(total_loss)
        val_loss_heatmap(losses[output_heatmap_idx])
        val_loss_paf2(losses[output_paf_idx])
        val_loss_paf1(losses[1])
        val_loss_paf0(losses[0])
        # x, y0, y1 = o_it.get_next()
        # losses, total_loss = val_one_step(model, x, y0, y1)
        # val_loss(total_loss)
        # val_loss_heatmap(losses[output_heatmap_idx])
        # val_loss_paf2(losses[output_paf_idx])
        # val_loss_paf1(losses[1])
        # val_loss_paf0(losses[0])
        # del losses, total_loss
    end_t = timer()
    tf.print("Epoch validing time: " + str(timedelta(seconds=end_t - start_t)))
    gc.collect()
    update_val_scalar(epoch)


@tf.function
def train_one_step(model, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        # y_pred = model.predict(x)
        losses = [eucl_loss(y_true[0], y_pred[0]),
                  eucl_loss(y_true[0], y_pred[1]),
                  eucl_loss(y_true[0], y_pred[2]),
                  eucl_loss(y_true[1], y_pred[3])
                  ]

        total_loss = tf.reduce_sum(losses)
        # del y_pred

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses, total_loss


# @profiler
def train(ds_train,  ds_val, model, optimizer, ckpt, manager, last_epoch, last_step, max_epochs,
          steps_per_epoch):
    resume = last_step != 0 and (steps_per_epoch - last_step) != 0
    start_epoch = last_epoch if resume else last_epoch + 1
    image_cnt= last_epoch * 17 + last_step // 80 + 1

    for epoch in range(start_epoch, max_epochs + 1, 1):
        # steps = 0
        start_t = timer()
        tf.print("Start to process epoch {}".format(epoch))

        if resume:
            train_idx = last_step + 1
            data_t_it = ds_train.skip(last_step)
            # data_t_it = ds_train.skip(last_step//2)
            # data_t_it2 = iter(ods_train.skip(last_step//2))
            tf.print(f"Skipping {last_step} steps")
            resume = False
        else:
            train_idx = 0
            data_t_it = ds_train
            # data_t_it2 = ods_train.make_initializable_iterator()

        # print(f"current iters {len(list(ds_train))}")
        for x, y0, y1 in data_t_it:
            train_idx += 1

            # train one step with profiler trace on in some steps -> run profiler too long will cause memory leak
            if train_idx == 500 and epoch % 5 == 1:
                tf.summary.trace_on(graph=True, profiler=True)
            losses, total_loss = train_one_step(model, optimizer, x, [y0, y1])
            #
            # ox, oy0, oy1 = data_t_it2.get_next()
            # losses2, total_loss2 = train_one_step(model, optimizer, ox, [oy0, oy1])
            # update_train_loss(total_loss=total_loss2, losses=losses2)

            if train_idx == 520 and epoch % 5 ==1:
                with graph_summary_writer.as_default():
                    tf.summary.trace_export(name="train_one_step_trace", step=0, profiler_outdir=graph_log_dir)

            # update all train loss summary
            update_train_loss(total_loss=total_loss, losses=losses)

            # update tensorboard scalar and image
            if train_idx % 20 == 0:
                tf.print('Epoch', epoch, f'Step {train_idx}/{steps_per_epoch}',
                         'Paf1', losses[0], 'Paf2', losses[1], 'Paf3', losses[2],
                         'Heatmap', losses[3], 'Total loss', total_loss)
                update_scalar(epoch, steps_per_epoch, train_idx)
                # reset train loss
                reset_train_loss()
            if train_idx % plot_update_steps == 0:
                # summary_step = (epoch * steps_per_epoch + train_idx) / plot_update_steps + 1
                update_plot(model, int(image_cnt))
                image_cnt+=1
            # save ckpt
            if train_idx % 300 == 0:
                save_ckpt(epoch, train_idx, ckpt, manager)
            # steps = train_idx

            if train_idx >= steps_per_epoch:
                break
        # steps =train_idx
        # train_idx = 0
        gc.collect()

        # save model
        tf.print("Complete epoch {}. Save weights...".format(epoch))
        model.save_weights(output_weights, overwrite=True)
        model.save(output_model, save_format='tf')
        save_ckpt(epoch, train_idx, ckpt, manager)

        # satisfy time
        end_t = timer()
        tf.print("Epoch training time: " + str(timedelta(seconds=end_t - start_t)))

        # final valid val-dataset
        val_one_epoch(ds_val, model, epoch)


# @profiler
if __name__ == '__main__':
    # loading datasets
    ds_val, ds_val_size = get_dataset(annot_path_val, img_dir_val, batch_size, strict=True, lmdb_file="None",use_o=False)
    ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size, lmdb_file="None", use_o=False)
    # ods_val, ods_val_size = get_dataset(annot_path_val, img_dir_val, batch_size, strict=True, lmdb_file="None",use_o=True)
    # ods_train, ods_train_size = get_dataset(annot_path_train, img_dir_train, batch_size, lmdb_file="None", use_o=True)
    print(f"Training samples: {ds_train_size} , Validation samples: {ds_val_size}, Batch_size: {batch_size}")

    # steps_per_epoch = ds_train_size // batch_size + ods_train_size // batch_size
    # steps_per_epoch_val = ds_val_size // batch_size + ods_val_size // batch_size
    steps_per_epoch = ds_train_size // batch_size
    steps_per_epoch_val = ds_val_size // batch_size

    # creating model, optimizers etc
    model, backend_layer = create_OpenposeSginglent()
    optimizer = Adam(lr)
    backend_layer.trainable = True
    model.compile(
        optimizer=optimizer
    )

    # loading previous state if required
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoints_folder, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    last_step = int(ckpt.step)
    last_epoch = int(ckpt.epoch)

    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
        print(f"Resumed from epoch {last_epoch}, step {last_step}")
    else:
        print(f"train new model without checkpoint")

    if last_epoch <= train_epoch:
        train(ds_train, ds_val, model, optimizer, ckpt, manager, last_epoch, last_step, train_epoch,
              steps_per_epoch)
        model.save_weights(output_weights + "-train", overwrite=True)
        model.save(output_model + "-train", save_format='tf')

    optimizer_fn = Adam(lr=1e-5)
    backend_layer.trainable = True
    model.compile(
        optimizer=optimizer_fn
        # run_eage
    )

    train(ds_train, ds_val, model, model, optimizer, ckpt, manager, last_epoch, last_step,
          fn_epoch + train_epoch,
          steps_per_epoch)

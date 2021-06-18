import tensorflow as tf

def create_tfrecord(ds,filepath):
    print(f"[DEBUG] write tfrecord: {filepath}-0.tfrecord")
    for i, _ in enumerate(ds.element_spec):
        ds_i = ds.map(lambda *args: args[i],num_parallel_calls=12).map(tf.io.serialize_tensor,num_parallel_calls=12)
        writer = tf.data.experimental.TFRecordWriter(f'{filepath}-{i}.tfrecord',compression_type='ZLIB')
        writer.write(ds_i)

def read_tfrecord(filepath):
    print(f"[DEBUG] read tfrecord: {filepath}-0.tfrecord")
    num_parts = 3
    data = []
    def read_map_fn(x):
        return tf.io.parse_tensor(x, tf.float32)

    for i in range(num_parts):
        data.append(tf.data.TFRecordDataset(f"{filepath}-{i}.tfrecord",buffer_size=20000,num_parallel_reads=12,compression_type='ZLIB').map(read_map_fn))
        rds = tf.data.Dataset.zip(tuple(data))

    return rds
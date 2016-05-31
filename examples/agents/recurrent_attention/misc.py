import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def print_flags():
    flags = FLAGS.__dict__['__flags']
    for f in flags:
        print f, flags[f]


def batch_add_images(images, b):
    images_shape = images.get_shape()
    batch_size = tf.shape(images)[0]

    TensorArray = tf.python.tensor_array_ops.TensorArray

    images_ta = TensorArray(dtype='float', size=batch_size).unpack(images)
    b_ta = TensorArray(dtype='float', size=batch_size).unpack(b)
    sum_ta = TensorArray(dtype='float', size=batch_size, name='sum_ta')

    cond = lambda i, *_: tf.less(i, batch_size)

    def body(i, sum_ta, images_ta, b_ta):
        image = images_ta.read(i)
        b_slice = b_ta.read(i)
        # broadcast add
        s = tf.expand_dims(tf.add(image, b_slice), 0)
        sum_ta = sum_ta.write(i, s)

        return (tf.add(i, 1), sum_ta, images_ta, b_ta)

    loop_vars = (tf.constant(0), sum_ta, images_ta, b_ta)

    final_loop_vars = tf.while_loop(cond, body, loop_vars)
    sum_ta = final_loop_vars[1]
    result = sum_ta.concat()

    result.set_shape(images_shape)

    return result


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    image = tf.constant(1.0, shape=(2, 1, 1, 3), dtype='float')

    x = [[1, 2, 3], [4, 5, 6]]
    h = tf.constant(x, shape=(2, 3), dtype='float')
    result_t = batch_add_images(image, h)
    assert result_t.get_shape().as_list() == [2, 1, 1, 3]

    result = sess.run(result_t)
    assert result.shape == (2, 1, 1, 3)
    assert np.allclose(result, [[[[2, 3, 4]]], [[[5, 6, 7]]]])

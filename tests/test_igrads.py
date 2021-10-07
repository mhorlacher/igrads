# %%
import numpy as np
import tensorflow as tf

import igrads
from tensorflow.python.ops.gen_batch_ops import batch

# %%
def _load_model(out_layer = None):
    model = tf.keras.models.load_model('test_data/test.model.h5')
    if out_layer is not None:
        o = model.layers[out_layer].output
        model = tf.keras.Model(inputs=model.inputs, outputs=[o])
    return model

def _make_input(batch_size=1):
    x = np.random.randint(4, size=101*batch_size)
    x = tf.one_hot(x, depth=4)
    x = tf.reshape(x, shape=(batch_size, 101, 4))
    return x

# %%
def test__compute_gradients():
    # arrange
    model = _load_model(-2)
    X = _make_input(batch_size=3)

    # act
    grads = igrads.igrads._compute_gradients(X, model)

    # assert
    tf.debugging.assert_equal(tf.shape(grads), (3, 101, 4))

# %%
def test__interpolate_inputs():

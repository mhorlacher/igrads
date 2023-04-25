# %%
import tensorflow as tf

# %%
from igrads.igrads import _compute_gradients, _apply_fn

# %%
@tf.function
def grad_x_input(inputs, model, **kwargs):
    inputs = tf.expand_dims(inputs, axis=0)
    grads = _compute_gradients(inputs, model, **kwargs)
    attribution = _apply_fn(lambda x: tf.squeeze(x * inputs), grads)
    return attribution
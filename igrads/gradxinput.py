# %%
import tensorflow as tf

# %%
from igrads.igrads import _compute_gradients

# %%
@tf.function
def grad_x_input(inputs, model, target_mask=None, postproc_fn=None):
    inputs = tf.expand_dims(inputs, axis=0)
    return tf.squeeze(_compute_gradients(inputs, model, target_mask=None, postproc_fn=None) * inputs)
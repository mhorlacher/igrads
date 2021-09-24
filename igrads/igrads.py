# %%
import tensorflow as tf

# %%
@tf.function
def _interpolate_inputs(inputs, baseline, steps=50):
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    delta = inputs - baseline
    interploated_inputs = tf.vectorized_map(lambda x: baseline + delta * x, alphas)
    return interploated_inputs

# %%
@tf.function
def _compute_gradients(inputs, model, target_mask=None):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        pred = model(inputs)
        if target_mask is not None:
            pred = tf.math.multiply(pred, target_mask)
    return tape.gradient(pred, inputs)

# %%
@tf.function
def _integral_approximation(gradients):
    # trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

# %%
#@tf.function
def integrated_gradients(inputs, model, target_mask=None, baseline=None, steps=50):
  # define zero baseline if no other baseline is specified
  if baseline is None:
    baseline = tf.zeros(tf.shape(inputs))

  # create interpolatioted inputs between inputs and baseline
  interpolated_inputs = _interpolate_inputs(inputs, baseline, steps)

  # compute gradients for interpolated inputs
  grads = _compute_gradients(interpolated_inputs, model, target_mask)

  # approximate the gradients integral
  integrated_grads = _integral_approximation(grads)

  # scale integrated gradients with respect to input.
  integrated_grads = (inputs - baseline) * integrated_grads

  return integrated_grads
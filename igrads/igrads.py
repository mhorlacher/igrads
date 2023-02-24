# %%
import numpy as np
import tensorflow as tf

# %%
# @tf.function
# def _interpolate_inputs(inputs, baseline, steps=50):
#     alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
#     delta = inputs - baseline
#     interploated_inputs = tf.vectorized_map(lambda x: baseline + delta * x, alphas)
#     return interploated_inputs
# %%
# @tf.function
def _interpolate(x, baseline, steps=50):
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    delta = x - baseline
    return tf.vectorized_map(lambda a: baseline + delta * a, alphas)

# %%
# @tf.function
def _nested_interpolate(x, baseline, steps=50):
    x_flattened = tf.nest.flatten(x)
    baseline_flattened = tf.nest.flatten(baseline)
    interpolated_flattened = [_interpolate(x, baseline, steps) for x, baseline in zip(x_flattened, baseline_flattened)]
    return tf.nest.pack_sequence_as(x, interpolated_flattened)

# %%
def _apply_fn(fn, inputs):
    if isinstance(inputs, tf.Tensor) or isinstance(inputs, np.ndarray):
        # if inputs is a tensor or array, apply function directly
        return fn(inputs)
    elif isinstance(inputs, list):
        # if inputs is a list, apply function to each element of inputs
        return [fn(x) for x in inputs]
    elif isinstance(inputs, dict):
        # if inputs is a dict, apply function to each value of inputs (preserve dict)
        return {key: fn(value) for key, value in inputs.items()}
    else:
        raise ValueError('Unsupported type: {}'.format(type(inputs)))

# %%
def _mask(x, mask):
    # apply mask to x (both are either tf.Tensor or np.ndarray)
    return tf.math.multiply(x, tf.stop_gradient(mask))

# %%
def _apply_mask(x, mask):
    if isinstance(mask, tf.Tensor) or isinstance(mask, np.ndarray):
        # if mask is a tensor, apply it to each element of x
        return _apply_fn(lambda y: _mask(y, mask), x)
    elif isinstance(mask, list):
        # if mask is a list, apply i'th element of mask to i'th element of x
        assert isinstance(x, list)
        assert len(x) == len(mask)
        return [_mask(x[i], mask[i]) for i in range(len(mask))]
    elif isinstance(mask, dict):
        # if mask is a dict, apply mask[key] to x[key]
        assert isinstance(x, dict)
        assert len(x) == len(mask)
        return {key: _mask(x[key], mask[key]) for key in mask}
    else:
        raise ValueError('Unsupported type: {}'.format(type(mask)))

# %%
def _compute_gradients(inputs, model, target_mask=None, postproc_fn=None):
    with tf.GradientTape(persistent=True) as tape:
        # watch gradients of inputs
        tape.watch(inputs)
        
        # run model predictions on inputs
        pred = model(inputs)

        if postproc_fn is not None:
            # optional: apply post-processing function (e.g. softmax on predicted logits)
            pred = _apply_fn(postproc_fn, pred)

        if target_mask is not None:
            # optional: apply mask, e.g. one-hot for classification
            pred = _apply_mask(pred, target_mask)

        # reduce_sum for each batch
        pred_sum = _apply_fn(lambda x: tf.reduce_sum(x, axis=tf.range(1, tf.rank(x))), pred)

    return _apply_fn(lambda x: tape.gradient(x, inputs), pred_sum)

# %%
#@tf.function
def _integral_approximation(gradients):
    # trapezoidal
    grads = _apply_fn(lambda x: (x[:-1] + x[1:]) / tf.constant(2.0), gradients)
    integrated_gradients = _apply_fn(lambda x: tf.math.reduce_mean(x, axis=0), grads)
    return integrated_gradients

# %%
def apply_fn_elementwise_structures(fun, struct1, struct2):
    tf.nest.assert_same_structure(struct1, struct2)

    struct1_flat, struct2_flat = tf.nest.flatten(struct1), tf.nest.flatten(struct2)
    results_flat = [fun(x, y) for x, y in zip(struct1_flat, struct2_flat)]

    return tf.nest.pack_sequence_as(struct1, results_flat)

# %%
#@tf.function
def integrated_gradients(inputs, baseline, model, target_mask=None, postproc_fn=None, steps=50):
    # TODO: define zero baseline if no other baseline is specified
    # if baseline is None:
    # baseline = tf.zeros(tf.shape(inputs))

    # create interpolatioted inputs between inputs and baseline
    interpolated_inputs = _nested_interpolate(inputs, baseline, steps)

    # compute gradients for interpolated inputs
    grads = _compute_gradients(interpolated_inputs, model, target_mask, postproc_fn)

    # approximate the gradients integral
    integrated_grads = _integral_approximation(grads)

    # scale integrated gradients with respect to input.
    # integrated_grads = _apply_fn(lambda x: (inputs - baseline) * x, integrated_grads)
    inputs_baseline_delta = apply_fn_elementwise_structures(lambda x, y: x-y, inputs, baseline)
    integrated_grads = apply_fn_elementwise_structures(lambda x, y: x*y, inputs_baseline_delta, integrated_grads)

    return integrated_grads
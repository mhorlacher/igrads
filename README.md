This is a Tensorflow 2.x implementation of the "Explainable AI" method *Integrated Gradients* of the publication [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) and [Integrated gradients | TensorFlow Core](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients) as a reference. 


*Installation:*
```
$ git clone https://github.com/mhorlacher/igrads.git
$ cd igrads
$ pip install .
```

*Usage:*
```python
X, Y = ... # load your data

model = ... # define your model

model.fit(...) # train your model

# load igrads
from igrads import integrated_gradients

# set inputs
inputs = X[0] # omit batch dimension

# define a baseline (and possibly a output mask, i.e. if attriutions should only be computed w.r.t. a single class)
baseline = ...
target_mask = ...

# compute input attributions
attribution = integrated_gradients(inputs, baseline, model, target_mask = target_mask)

# 'attribution' has the same shape as the input. 
```


*Notes / Caution:*
- Package was test with Tensorflow version 2.4.1 but should work with other versions of the 2.x major release
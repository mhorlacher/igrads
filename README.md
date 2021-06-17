This is a Tensorflow 2.x implementation of the "Explainable AI" method *Integrated Gradients* of the publication [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365). 


Installation:
```
$ git clone https://github.com/mhorlacher/igrads.git
$ cd igrads
$ pip install .
```

Usage:
```python
X, Y = ... # load your data

model = ... # define your model

model.fit(...) # train your model

# compute input attributions
from igrads import integrated_gradients

attribution = integrated_gradients(X[0], model)

# 'attribution' has the same shape as the input. 
```
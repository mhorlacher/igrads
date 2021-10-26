# %%
import tensorflow as tf

# %%
from igrads import integrated_gradients

# %%
class IGExplainer(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(IGExplainer, self).__init__()
        self.model = model
        self.kwargs = kwargs
    
    def explain_step(self, data, **kwargs):
        inputs, y = data
        return integrated_gradients(inputs, self.model, **kwargs)

    def predict_step(self, data):
        return self.explain_step(data=data, **self.kwargs)
    
    def __call__(self, inputs, **kwargs):
        raise NotImplementedError('IGExplain is a wrapper for interpretation only.')
    
    def explain(self, data):
        return self.predict(data)

    def call(self, *args, **kwargs):
        raise NotImplementedError('IGExplain is a wrapper for interpretation only.')

    def fit(self, *args, **kwargs):
        raise NotImplementedError('IGExplain is a wrapper for interpretation only.')

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError('IGExplain is a wrapper for interpretation only.')

    def compile(self, *args, **kwargs):
        raise NotImplementedError('IGExplain is a wrapper for interpretation only.')
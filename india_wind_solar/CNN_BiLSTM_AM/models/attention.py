"""Attention mechanism implementation for the CNN-BiLSTM-AM model."""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import backend as K

class AttentionLayer(Layer):
    """
    Attention layer for sequence models.
    This layer computes a weighted sum of the input sequence based on computed attention weights.
    """
    
    def __init__(self, units=128, **kwargs):
        self.units = units
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create trainable weights
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(name='attention_context_vector',
                                 shape=(self.units, 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        # uit shape: (batch_size, time_steps, units)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        
        # ait shape: (batch_size, time_steps, 1)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, axis=-1)
        
        # Convert to softmax weights
        ait = K.softmax(ait)
        
        # Expand to match dims for multiplication
        ait_expanded = K.expand_dims(ait, axis=-1)
        
        # Weighted sum (context vector)
        context = x * ait_expanded
        context = K.sum(context, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'units': self.units
        })
        return config
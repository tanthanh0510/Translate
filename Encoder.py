from EncoderLayer import EncoderLayer
import tensorflow as tf
import numpy as np
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    def get_angles(pos, i, d_model):
      angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
      return pos * angle_rates
    
    self.dropout = tf.keras.layers.Dropout(rate)
    def positional_encoding(position, d_model):
          angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                  np.arange(d_model)[np.newaxis, :],
                                  d_model)
          # apply sin to even indices in the array; 2i
          angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

          # apply cos to odd indices in the array; 2i+1
          angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

          pos_encoding = angle_rads[np.newaxis, ...]

          return tf.cast(pos_encoding, dtype=tf.float32)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)
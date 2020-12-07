import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from MultiHeadAttention import MultiHeadAttention
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from Encoder import Encoder
from Decoder import Decoder
from Transformer import Transformer
from CustomSchedule import CustomSchedule


class Model_translater():
    def __init__(self):
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer_en')
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer_vn')
        self.MAX_LENGTH = 400
        self.num_layers = 4
        self.d_model = 128
        self.dff = 512
        self.num_heads = 8
        self.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.tokenizer_en.vocab_size + 2
        self.dropout_rate = 0.1
        self.learning_rate = CustomSchedule(self.d_model)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, 
                                                    epsilon=1e-9)    
            
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')                        
        self.transformer = Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                                self.input_vocab_size, self.target_vocab_size, 
                                pe_input=self.input_vocab_size, 
                                pe_target=self.target_vocab_size,
                                rate=self.dropout_rate)
        self.checkpoint_path = "./checkpoints/train"

        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
    def get_angles(self,pos, i,d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        return pos * angle_rates
    def create_padding_mask(self,seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions to add the padding to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    def create_look_ahead_mask(self,size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    
    def scaled_dot_product_attention(self,q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights
    def create_masks(self,inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
        
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask
    def evaluate(self,inp_sentence):
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size + 1]
    
        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)
    
        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
    
        for i in range(self.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                encoder_input, output)
    
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.tokenizer_en.vocab_size+1:
            
                return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0), attention_weights    
    def translate(self,sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)    
        predicted_sentence = self.tokenizer_en.decode([i for i in result 
                                                if i < self.tokenizer_en.vocab_size])  
        return sentence,predicted_sentence

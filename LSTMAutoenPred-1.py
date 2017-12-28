
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import numpy as np

"""
Future : Modularization
"""

class LSTMAutoencoder(object):
  """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)

  Usage:
    ae = LSTMAutoencoder(hidden_num, inputs)
    sess.run(ae.train)
  """

  def __init__(self, hidden_num, inputs, predinputs, 
    cell=None, optimizer=None, reverse=True, 
    decode_without_input=True):

    self.batch_num = list(inputs[0].shape)[0]
    self.elem_num = list(inputs[0].shape)[1]

    if cell is None:
      self._enc_cell = BasicLSTMCell(hidden_num, state_is_tuple=True)
      self._dec_cell = BasicLSTMCell(hidden_num, state_is_tuple=True)
      self._pred_cell = BasicLSTMCell(hidden_num, state_is_tuple=True)
      
    else :
      self._enc_cell = cell
      self._dec_cell = cell
      self._pred_cell = cell

    with tf.variable_scope('encoder'):
        enc_state = (tf.zeros([self.batch_num,hidden_num]), tf.zeros([self.batch_num,hidden_num]))
        self.z_codes = []
        inputs_tf=tf.convert_to_tensor(inputs)
        for step in range(len(inputs)):
          enc_output, enc_state = self._enc_cell(inputs_tf[step], enc_state)
          self.z_codes.append(enc_output)
        self.enc_state = enc_state
        print(self.z_codes)

    with tf.variable_scope('decoder') as vs:
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], stddev=0.1, dtype=tf.float32),
        name="dec_weight")
      dec_bias_ = tf.Variable(
        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
        name="dec_bias")   

      if decode_without_input:
        dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                      for _ in range(len(inputs))]
        dec_state = self.enc_state
        dec_outputs = []

        for step in range(len(inputs)):
          dec_output, dec_state = self._dec_cell(dec_inputs[step], dec_state)
          dec_outputs.append(dec_output)

        if reverse:
          dec_outputs = dec_outputs[::-1]
        dec_output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])
        dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [self.batch_num,1,1])
        self.output_ = tf.matmul(dec_output_, dec_weight_) + dec_bias_

      else : 
        dec_state = self.enc_state
        dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_outputs = []
        for step in range(len(inputs)):
          if step>0: vs.reuse_variables()
          dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
          dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
          dec_outputs.append(dec_input_)
        if reverse:
          dec_outputs = dec_outputs[::-1]
        self.output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])

    with tf.variable_scope('predictor') as vs1:
      # Predictor Weight and bias      
      pred_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], stddev=0.1, dtype=tf.float32),
        name="pred_weight")
      pred_bias_ = tf.Variable(
        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
        name="pred_bias")      

      # Predictor Construction
      pred_state = self.enc_state
      pred_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
      
      pred_outputs = []
      for step in range(len(inputs)):
        if step>0: vs1.reuse_variables()
        pred_input_, pred_state = self._pred_cell(pred_input_, pred_state)
        pred_input_ = tf.matmul(pred_input_, pred_weight_) + pred_bias_
        pred_outputs.append(pred_input_)
      self.predoutput_ = tf.transpose(tf.stack(pred_outputs), [1,0,2])       
      
    self.predinput_ = tf.transpose(tf.stack(predinputs), [1,0,2])     
    self.input_ = tf.transpose(tf.stack(inputs), [1,0,2])
    self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_)) \
                + tf.reduce_mean(tf.square(self.predinput_ - self.predoutput_))

    if optimizer is None :
      self.train = tf.train.AdamOptimizer().minimize(self.loss)
    else :
      self.train = optimizer.minimize(self.loss)

inputs=np.array([[[1.],[4.]],[[2.],[5.]],[[3.],[6.]]],dtype=np.float32)
testX=np.array([[[4.],[5.]],[[5.],[6.]],[[6.],[7.]]],dtype=np.float32)

predinputs=np.array([[[-2.],[1.]],[[-1.],[2.]],[[0.],[3.]]],dtype=np.float32)
predtestX=np.array([[[1.],[2.]],[[2.],[3.]],[[3.],[4.]]],dtype=np.float32)

ae = LSTMAutoencoder(4, inputs, predinputs)
with tf.Session() as sess:
       init = tf.global_variables_initializer()
       sess.run(init)

# Training step
       for i in range(5000):
          _, step_loss = sess.run([ae.train, ae.loss])
          if i % 500 == 0:
              print("[step: {}] loss: {}".format(i, step_loss))

# Testing step
       ae.inputs = testX
       ae.predinputs = predtestX
       
       test_predict = sess.run(ae.predoutput_) 
       tg = np.transpose(predtestX, [1,0,2])    
       t_loss = np.mean(np.square(tg - test_predict))
       print("Test_loss: {}".format(t_loss))
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
    cell=None, optimizer=None, reverse=True, decode_without_input=True):      
      
    
    self.batch_num = inputs[0].shape[0]
    self.elem_num  = inputs[0].shape[1]
    

    if cell is None:
      self._enc_cell  = BasicLSTMCell(hidden_num)
      self._dec_cell  = BasicLSTMCell(hidden_num)
      self._pred_cell = BasicLSTMCell(hidden_num)
      
    else :
      self._enc_cell = cell
      self._dec_cell = cell
      self._pred_cell = cell

    with tf.variable_scope('encoder'):
        enc_state = (tf.zeros([self.batch_num,hidden_num]), tf.zeros([self.batch_num,hidden_num])) # (cell state, hidden state) tuple
        #self.z_codes = []
        inputs_tf=tf.convert_to_tensor(inputs)
        for step in range(len(inputs)):
          enc_output, enc_state = self._enc_cell.call(inputs_tf[step], enc_state) # .call(input, state) : calls LSTM, otherwise RNN.
          #self.z_codes.append(enc_output)
        self.enc_state = enc_state
        #tf.Print(self.enc_state, [self.enc_state])

    with tf.variable_scope('decoder') as vs:
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, self.elem_num], stddev=0.1, dtype=tf.float32),
        name="dec_weight")
      dec_bias_ = tf.Variable(tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32),
        name="dec_bias")

      if decode_without_input:
        dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                      for _ in range(len(inputs))]
        dec_state = self.enc_state
        dec_outputs = []

        for step in range(len(inputs)):
          dec_output, dec_state = self._dec_cell.call(dec_inputs[step], dec_state)
          dec_outputs.append(dec_output)
          # output shape : (2,4) = (batch_size, hidden_state_size)
          
        if reverse:
          dec_outputs = dec_outputs[::-1]
        #dec_output_ = tf.transpose(dec_outputs, [0,1,2]) # transpose from (3,2,4) into (3,2,4)
        dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [len(inputs),1,1])  # (3,4,1)
        #print(dec_weight_.shape)
        self.output_ = tf.matmul(dec_outputs, dec_weight_) + dec_bias_ # (3,2,1)
        #print(len(dec_outputs), dec_outputs[0].shape)
        self.dec_outputs = dec_outputs

      else :        
        dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_state = self.enc_state
        dec_outputs = []
        for step in range(len(inputs)):
          if step>0: vs.reuse_variables()
          dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
          dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
          dec_outputs.append(dec_input_)
        if reverse:
          dec_outputs = dec_outputs[::-1]
        #self.output_ = tf.transpose(dec_outputs, [0,1,2])  # (3,2,1) = (time,batch,hidden)
        self.output_ = dec_outputs
        
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
        pred_input_, pred_state = self._pred_cell.call(pred_input_, pred_state)
        # pred_input_ shape : (2,4) = (batch, hidden)
        pred_input_ = tf.matmul(pred_input_, pred_weight_) + pred_bias_
        # pred_input_ shape : (2,1) = (batch, element)
        pred_outputs.append(pred_input_)
        
      #self.predoutput_ = tf.transpose(pred_outputs, [0,1,2]) # (3,2,1) = (time, batch, element) 
      self.predoutputs = tf.convert_to_tensor(pred_outputs)  
    
    #self.predinput_ = tf.transpose(predinputs, [0,1,2])     # transpose from (3,2,1) into (3,2,1)
    #self.input_ = tf.transpose(tf.stack(inputs), [0,1,2])
    self.loss = tf.reduce_mean(tf.square(inputs - self.output_)) \
                + tf.reduce_mean(tf.square(predinputs - self.predoutputs))

    if optimizer is None :
      self.train = tf.train.AdamOptimizer().minimize(self.loss)
    else :
      self.train = optimizer.minimize(self.loss)
    
#inputs=np.array([[[1],[2]],[[2],[3]],[[3],[4]]],dtype=np.float32)
#predinputs=np.array([[[4],[5]],[[5],[6]],[[6],[7]]],dtype=np.float32)
inputs=[]
predinputs=[]
for i in range(-50,50):
    input_ = [[i],[i+1],[i+2]]
    predinput_ = [[i+3],[i+4],[i+5]]
    inputs.append(input_)
    predinputs.append(predinput_)
inputs = np.array(inputs, dtype=np.float32,).transpose([1,0,2])
predinputs = np.array(predinputs, dtype=np.float32,).transpose([1,0,2])

testX=np.array([[[3]],[[4]],[[5]]],dtype=np.float32)
predtestX=np.array([[[6]],[[7]],[[8]]],dtype=np.float32)

ae = LSTMAutoencoder(10, inputs, predinputs)
with tf.Session() as sess:
       init = tf.global_variables_initializer()
       sess.run(init)

# Training step
       for i in range(20000):
          _, step_loss = sess.run([ae.train, ae.loss])
          if i % 500 == 0:
              print("[step: {}] loss: {}".format(i, step_loss))

# Testing step
       ae.inputs = testX
       ae.predinputs = predtestX
       
       test_decoder = sess.run(ae.output_)
       test_predict = sess.run(ae.predoutputs) 
       #tg = np.transpose(predtestX, [0,1,2])  
       loss_decoder = np.mean(np.square(testX - test_decoder))
       loss_predict = np.mean(np.square(predtestX - test_predict))
       print("Decode_test_loss: {}".format(loss_decoder))
       print("Predict_test_loss: {}".format(loss_predict))
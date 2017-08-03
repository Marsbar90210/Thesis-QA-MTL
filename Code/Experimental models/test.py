from keras.models import Model
from keras import backend as K
from keras.layers import Input, Activation
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)

var = tf.placeholder(tf.float32, (None, 1), name="test")
input = Input(tensor=var)
q_emb = tf.transpose(input, [0,1])
pred = Activation("softmax")(q_emb)
Model(inputs=input, outputs=pred)
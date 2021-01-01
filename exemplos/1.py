# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:39:56 2020

Backpropagation algorithm


@author: ELETRON
"""
#####################TENSOR BOARD ######################################
# import tensorflow as tf
# import tensorboard


# a = tf.constant(2, dtype=tf.float32, name="tensor_a")
# b = tf.constant(4, dtype=tf.float32, name="tensor_b")
# c = tf.add(a,b)

# file_name = '/tensorFlow'
# writer = tf.summary.create_file_writer(file_name)


# with writer.as_default():
#   for step in range(100):
#     # other model code would go here
#     tf.summary.scalar("my_metric", 0.5, step=step)
#     writer.flush()
    
#########################################################################

### imports
import tensorflow.compat.v1 as tf



tf.disable_v2_behavior()
### constant data
x  = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
y_ = [[0.],[0.],[1.],[1.]]

### induction
# 1x2 input -> 2x3 hidden sigmoid -> 3x1 sigmoid output

# Layer 0 = the x2 inputs
x0 = tf.constant( x  , dtype=tf.float32 )
y0 = tf.constant( y_ , dtype=tf.float32 )

# Layer 1 = the 2x3 hidden sigmoid
m1 = tf.Variable( tf.random.uniform( [2,3] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b1 = tf.Variable( tf.random.uniform( [3]   , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
h1 = tf.sigmoid( tf.matmul( x0,m1 ) + b1 )

# Layer 2 = the 3x1 sigmoid output
m2 = tf.Variable( tf.random.uniform( [3,1] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b2 = tf.Variable( tf.random.uniform( [1]   , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
y_out = tf.sigmoid( tf.matmul( h1,m2 ) + b2 )


### loss
# loss : sum of the squares of y0 - y_out
loss = tf.reduce_sum( tf.square( y0 - y_out ) )

# training step : gradient decent (1.0) to minimize loss
train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
# tf.train.GradientDescentOptimizer(0.5)


### training
# run 500 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:
  sess.run( tf.global_variables_initializer() )
  for step in range(500) :
    sess.run(train)

  results = sess.run([m1,b1,m2,b2,y_out,loss])
  labels  = "m1,b1,m2,b2,y_out,loss".split(",")
  for label,result in zip(*(labels,results)) :
    print ("")
    print (label)
    print (result)
sess.close()
print ("")
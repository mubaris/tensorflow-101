import tensorflow as tf

# Declare placeholder with datatype
x = tf.placeholder(tf.float32)

# You can also define constant with specified datatype
a = tf.constant(32, dtype=tf.float32)
y = tf.placeholder(tf.float32)

z = a*x + y*y

sess = tf.Session()

print(sess.run(z, {x: 2, y: 4})) # 80.0
print(sess.run(z, {x: [1, 2, 3], y: [2, 3, 4]})) # [36. 73. 112.]

# Define Variables
W = tf.Variable([.25], dtype=tf.float32)
b = tf.Variable([-.64], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Initialize 
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [4, 5, 1, 8]}))
# [ 0.36000001  0.61000001 -0.38999999  1.36000001]

sess.close()
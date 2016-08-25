import tensorflow as tf
import draft_tf_class


model = draft_tf_class.tfmodel()

sess = tf.Session()

print sess.run(model.z)

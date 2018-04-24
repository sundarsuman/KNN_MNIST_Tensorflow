import tensorflow as tf 
import numpy as np 

#Import data
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("mnist_data/", one_hot = True)

training_digits, training_label = mnist.train.next_batch(20000)

test_digits, test_label = mnist.test.next_batch(1000)

training_digits_pl = tf.placeholder("float", [None,784])
test_digits_pl = tf.placeholder("float", [784])


#Calculate L1 distance 
l1_distance = tf.abs(tf.add(training_digits_pl,tf.negative(test_digits_pl)))

distance = tf.reduce_sum(l1_distance,axis=1)

pred = tf.arg_min(distance,0)

accuracy = 0

with tf.Session() as sess:
	# print sess.run(pred,feed_dict={training_digits_pl:training_digits, test_digits_pl: test_digits[100,:]})

	for i in range(len(test_digits)):
		nn_index = sess.run(pred,feed_dict={training_digits_pl:training_digits, test_digits_pl: test_digits[i,:]})
		# print ("Test", i, "Prediction:", np.argmax(training_label[nn_index]), "True Label: ", np.argmax(test_label[i]))

		if np.argmax(training_label[nn_index]) == np.argmax(test_label[i]): accuracy += 1./len(test_digits)

		if np.argmax(training_label[nn_index]) != np.argmax(test_label[i]): print ("Test: ", i, "Prediction: ", np.argmax(training_label[nn_index]), "True label: ",np.argmax(test_label[i]))
	# print "Accuracy: ", accuracy

		
	sess.close()







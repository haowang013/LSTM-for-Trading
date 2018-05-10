# -*- coding: utf-8 -*-
"""
author: Wang
"""
import lstm
import pandas as pd
import numpy as np
import tensorflow as tf

hidden_rnn_layer = 15
learning_rate = 0.005
input_size = 7
output_size = 1
batch_size = 10
time_step = 5
LSTM_layer_number = 2

TestingFile = "E:\\py_script\\ivx_rnn\\test.csv"
def LoadData(number):
	rural_testing_data = pd.read_csv(TestingFile)
	testing_data = rural_testing_data.iloc[:number,0:7]
	for i in range(7):
		testing_data[[str(i+1)]] = (np.array(testing_data[[str(i+1)]] - np.mean(np.array(testing_data[[str(i+1)]]))))/np.std(np.array(testing_data[[str(i+1)]])) 
	testing_data = testing_data.values
    #testing_label = testing_label.values
	size = (len(testing_data)+time_step-1)//time_step
	test_x = []
	for i in range(size-1):
		x = testing_data[i*time_step:(i+1)*time_step,:7]
        #y = testing_label[i*time_step:(i+1)*time_step,]
		test_x.append(x.tolist())
        #test_y.extend(y)
	test_x.append((testing_data[(i+1)*time_step:,:7]).tolist())
    #test_y.extend((testing_label[(i+1)*time_step:,]).tolist())
	test_x = np.array(test_x)
    #test_y = np.array(test_y)
	return test_x


def Predict(number):
	X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
	test_x = LoadData(number)
	with tf.variable_scope('sec_lstm',reuse = True):
		output_data,_ = lstm.LSTM_model(X)

	init_op = tf.global_variables_initializer()
	model_dir = "E:\\py_script\\ivx_rnn"

	saver = tf.train.Saver(tf.global_variables())
	model_file = tf.train.latest_checkpoint(model_dir)

	with tf.Session() as sess:
		saver.restore(sess,model_file)
        
		yhat = []
		for i in range(len(test_x)-1):
			p = sess.run(output_data,feed_dict={X:[test_x[i]]})
			yhat.append(tf.reshape(p,[-1]))

        #y = tf.reshape(test_y,[-1])
		yhat = tf.reshape(yhat,[1,-1])
		array_yhat = np.array(yhat.eval(session=sess))

	array_yhat1 = []
	for i in range(np.shape(array_yhat)[0]):
		if array_yhat[0,i] > 0:
			array_yhat1.append(1)
		elif array_yhat[0,i] < 0:
			array_yhat1.append(-1)
	array_yhat1 = np.reshape(np.array(array_yhat1),[1,-1])

	return array_yhat1[0,-1]

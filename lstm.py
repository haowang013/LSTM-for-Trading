import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import time


# LSTM模型的参数设置
hidden_rnn_layer = 10
alpha = 0.005
input_size = 7
output_size = 1
batch_size = 6

time_step = 4

# 设置weight和bias以及forget gate
forget_rate = {
    'input_keep_prob':tf.Variable(tf.constant(0.9)),
    'output_keep_prob':tf.Variable(tf.constant(0.97))
}

weights = {
    'in':tf.Variable(tf.random_normal([hidden_rnn_layer,input_size])),
    'out':tf.Variable(tf.random_normal([output_size,hidden_rnn_layer]))
}

bias = {
    'in':tf.Variable(tf.constant(0.0,shape=[hidden_rnn_layer,1])),
    'out':tf.Variable(tf.constant(0.0))
}


TrainingFile = "E:\\py_script\\ivx_rnn\\train.csv"
TestingFile = "E:\\py_script\\ivx_rnn\\test.csv"

def LoadTrainingData():
    rural_training_data = pd.read_csv(TrainingFile)
    training_data = rural_training_data.iloc[:,0:7]
    training_label = rural_training_data.iloc[:-1]
    for i in range(7):
        training_data[[str(i+1)]] = (np.array(training_data[[str(i+1)]]) - np.mean(np.array(training_data[[str(i+1)]])))/np.std(np.array(training_data[[str(i+1)]]))
    training_data = training_data.values
    training_label = training_label.values
    train_x, train_y = [],[]
    batch_index = []
    for i in range(len(training_data)-time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = training_data[i:i+time_step,:7]
        y = training_label[i:i+time_step,-1,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(training_data)-time_step))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return batch_index, train_x, train_y


def LoadTestingData():
    rural_testing_data = pd.read_csv(TestingFile)
    testing_data = rural_testing_data.iloc[:,0:7]
    testing_label = rural_testing_data.iloc[:,-1]
    for i in range(7):
        testing_data[[str(i+1)]] = (np.array(testing_data[[str(i+1)]] - np.mean(np.array(testing_data[[str(i+1)]]))))/np.std(np.array(testing_data[[str(i+1)]])) 
    testing_data = testing_data.values
    testing_label = testing_label.values
    train_x, train_y = [],[]
    size = (len(testing_data)+time_step-1)//time_step
    test_x, test_y = [],[]
    for i in range(size-1):
        x = testing_data[i*time_step:(i+1)*time_step,:7]
        y = testing_label[i*time_step:(i+1)*time_step,]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((testing_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((testing_label[(i+1)*time_step:,]).tolist())
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return test_x,test_y
    
    
## 构建LSTM模型
def LSTM_model(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = bias['in']
    input_data = tf.reshape(X,[input_size,-1])
    input_lstm = tf.matmul(w_in,tf.cast(input_data,tf.float32)) + b_in
    input_lstm = tf.reshape(input_lstm,[-1,time_step,hidden_rnn_layer])
    input_lstm = tf.nn.leaky_relu(input_lstm)
    
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_rnn_layer,forget_bias=0.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=forget_rate['input_keep_prob'],
                                         output_keep_prob=forget_rate['output_keep_prob'])
    
    init_state = cell.zero_state(batch_size,dtype=tf.float32)
    
    output_lstm, final_state = tf.nn.dynamic_rnn(cell,input_lstm, initial_state=init_state, dtype=tf.float32)
    output_lstm = tf.reshape(output_lstm,[hidden_rnn_layer,-1])
    w_out = weights['out']
    b_out = bias['out']
    
    output_data = tf.matmul(w_out,output_lstm) + b_out
    output_data = tf.tanh(output_data)

    return output_data,final_state
    
    
def TrainLSTM():
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index,train_x,train_y = LoadTrainingData()
    with tf.variable_scope('sec_lstm'):
        output_data,_ = LSTM_model(X)
        
    Loss = tf.reduce_mean(tf.square(tf.reshape(output_data,[-1])-tf.reshape(Y,[-1])))
    
    train_op = tf.train.AdamOptimizer(alpha).minimize(Loss)
    
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    init_op = tf.global_variables_initializer()
    
    loss_value = []
    iteration = []
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(4000):
            for j in range(len(batch_index)-1):
                loss_,_ = sess.run([Loss,train_op],feed_dict={X:train_x[batch_index[j]:batch_index[j+1]],Y:train_y[batch_index[j]:batch_index[j+1]]})
            print("number of iteration:",i,"loss function is:",loss_)
            iteration.append(i)
            loss_value.append(loss_)
        print("model_save: ",saver.save(sess,'E:\\py_script\\ivx_rnn\\lstm_single'))
        plt.plot(iteration,loss_value,label='Loss Function')
        plt.legend()
        plt.show()
        print("finish the training process")   
        
        
def LSTMPredict():
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    test_x, test_y = LoadTestingData()
    with tf.variable_scope('sec_lstm'):
        output_data,_ = LSTM_model(X)
    
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
        y = tf.reshape(test_y,[-1])
    
    return y,yhat
    

def PreAccuracy(y,yhat):
    y = tf.reshape(y,[-1,1])
    yhat = tf.reshape(yhat,[-1,1])
    with tf.Session() as sess:
        array_y = np.array(y.eval(session=sess))
        array_yhat = np.array(yhat.eval(session=sess))
    
    array_yhat1 = np.array([1 if i>0 else -1 for i in array_yhat])
    count_array1 = array_yhat1 - array_y
    total_accuracy = (np.sum(count_array1==0))/len(array_y)
    
   # accuracy of 1
    array_yhat2 = np.array([1 if i>0 else -100 for i in array_yhat]) 
    count_array2 = array_yhat2 - array_y
    plus1_accuracy = (np.sum(count_array2==0))/len(array_y)
     
    # accuracy of -1
    array_yhat3 = np.array([100 if i>0 else -1 for i in array_yhat])
    count_array3 = array_yhat3 - array_y
    minus1_accuracy = (np.sum(count_array3==0))/len(array_y)
    
    return total_accuracy,plus1_accuracy,minus1_accuracy

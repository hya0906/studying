"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import losses
from TIMNET_dist import TIMNET
global count
count = 0
batchsize = 48
epch = 1

def kl_ce_loss(t_pred, alpha):
    def loss(y_true, y_pred):
        global count, batchsize, epch
        print(y_true.shape[0], y_pred.shape, t_pred.shape)
        print(batchsize * count, batchsize * (count + 1))

        # Extract the relevant batch indices for t_pred and y_pred
        t_pred_batch = t_pred[batchsize * count:batchsize * (count + 1)]
        #y_pred_batch = y_pred[batchsize * count:batchsize * (count + 1)]
        print("t_pred",t_pred_batch.shape)
        # KL divergence 계산
        kl_loss = tf.reduce_sum(t_pred_batch * tf.math.log(t_pred_batch / (y_pred + tf.keras.backend.epsilon())), axis=-1)
        print("kl_loss", kl_loss)
        # Cross entropy 계산
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # KL divergence와 Cross entropy를 합친 최종 loss 계산
        total_loss = alpha * kl_loss + (1-alpha) * ce_loss

        # Update batch index
        count += 1
        count %= (y_true.shape[0] // batchsize)
        return total_loss

    return loss

'''
class kl_ce_loss(losses.Loss):
    def call(self, y_true, y_pred, t_pred):
        # KL divergence 계산
        kl_loss = tf.reduce_sum(t_pred * tf.math.log(t_pred / (y_pred + tf.keras.backend.epsilon())), axis=-1)

        # Cross entropy 계산
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # KL divergence와 Cross entropy를 합친 최종 loss 계산
        total_loss = kl_loss + ce_loss
        return total_loss
'''
'''
def kl_ce_loss(y_true, y_pred):
    # KL divergence 계산
    kl_loss = tf.reduce_sum(y_true * tf.math.log(y_true / (y_pred + tf.keras.backend.epsilon())), axis=-1)

    # Cross entropy 계산
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # KL divergence와 Cross entropy를 합친 최종 loss 계산
    total_loss = kl_loss + ce_loss

    return total_loss
'''

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        #print("===call===")
        #print("!",x.shape)
        tempx = tf.transpose(x,[0,2,1])  #(None, 39, 8) 그냥 transpose
        #print("!!", tempx.shape)
        x = K.dot(tempx,self.kernel)  #(None, 39, 1)
        #print("!!!",x.shape)
        x = tf.squeeze(x,axis=-1)  #(None, 39)
        #print("!!!!",x.shape)
        #print("===call end===")
        return  x

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

def softmax(x, axis=-1):
    ex = K.exp((x - K.max(x, axis=axis, keepdims=True))/3.5)
    return ex/K.sum(ex, axis=axis, keepdims=True)

class TIMNET_Model(Common_Model): #tf.keras.Model로 변경가능
    def __init__(self, dilation_size, filter_size, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.dilation_size = dilation_size
        self.filter_size = filter_size
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        #print("TIMNET MODEL SHAPE:",input_shape)
        self.create_model()
        #self.create_model2()

    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=65,
                                kernel_size=self.args.kernel_size,
                                nb_stacks=self.args.stack_size,
                                dilations=8,
                                dropout_rate=self.args.dropout,
                                activation = self.args.activation,
                                return_sequences=True,
                                name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.teacher_model = Model(inputs = self.inputs, outputs = self.predictions)

        self.teacher_model.compile(loss = "categorical_crossentropy",
                           optimizer =Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics = ['accuracy'])
        print("Temporal create succes!")

    def create_model2(self):
        self.inputs = Input(shape=(self.data_shape[0], self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.filter_size,
                                     kernel_size=self.args.kernel_size,
                                     nb_stacks=self.args.stack_size,
                                     dilations=self.dilation_size,
                                     dropout_rate=self.args.dropout,
                                     activation=self.args.activation,
                                     return_sequences=True,
                                     name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.student_model = Model(inputs=self.inputs, outputs=self.predictions)
        #####
        loss_function = tf.keras.losses.KLDivergence()

        #self.student_model.compile(loss=kl_ce_loss,
        #                           optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1,
        #                                          beta_2=self.args.beta2, epsilon=1e-8),
        #                           metrics=['accuracy'])

        #####
        #self.student_model.compile(loss="categorical_crossentropy",
        #                   optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2,
        #                                  epsilon=1e-8),
        #                   metrics=['accuracy'])
        print("Temporal create succes!")


    def test(self, x, y, path):
        i=1
        scores = np.zeros(10)
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        graphs = []
        result_list = []
        for kfold, (train, test) in enumerate(kfold.split(x, y)):
            self.create_model2()
            print(self.teacher_model, self.student_model)
            weight_path = f".\\RAVDESS_mfcc_60_46_main2_2023-06-03_00-53-18_92.847\\{str(self.args.split_fold)}-fold_weights_best_{str(i)}.hdf5"
            #weight_path = path + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".hdf5"

            self.teacher_model.fit(x[train], y[train],validation_data=(x[test],  y[test]), batch_size=48, epochs = 0,verbose=1)
            print(weight_path)
            self.teacher_model.load_weights(weight_path)
            self.teacher_model.evaluate(x[test], y[test])
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            X_train_kf, X_test_kf = x[train], x[test]
            #print(teacher_model)
            y_train_kf = self.teacher_model.predict(x[train])
            #y_train_kf = smooth_labels(y_train_kf, 0.1)
            self.student_model.compile(loss=kl_ce_loss(y_train_kf, 0.1),
                                       optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1,
                                                      beta_2=self.args.beta2, epsilon=1e-8),
                                       metrics=['accuracy'])


            y[train] = smooth_labels(y[train], 0.1)
            weight_path_=f"./kl_ce0.1/{str(self.args.split_fold)}-fold_weights_best_{str(i)}.hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path_, monitor='val_accuracy', verbose=1,
                                                   save_weights_only=True, save_best_only=True, mode='max')
            h = self.student_model.fit(X_train_kf, y[train], validation_data=(x[test],  y[test]), batch_size=48, epochs=self.args.epoch, verbose=1,callbacks=[checkpoint])
            y_pred = self.student_model.predict(X_test_kf, batch_size=48)
            print(y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            print(type(y_pred))
            print(y_pred)
            print(type(self.teacher_model.predict(x[test])))
            print(self.teacher_model.predict(x[test], batch_size=48))

            self.student_model.load_weights(weight_path_)
            best_eva_list = self.student_model.evaluate(x[test], y[test], batch_size=48)
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            result_list.append(best_eva_list[1])
            #print("avg_accuracy",avg_accuracy)
            print(str(i) + '_Model evaluation: ', best_eva_list, "   Now ACC:",str(round(avg_accuracy * 10000) / 100 / i))

            score = accuracy_score(np.argmax(self.teacher_model.predict(x[test]), axis=1), y_pred)
            print(score)
            graphs.append(h)

            #scores[i] = score
            print("result_list", result_list)
            #print(f"Fold: {i + 1}, accuracy: {score}")
            i+=1
        for graph in graphs:
            history_dict = graph.history
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            epochs = range(1, len(loss) + 1)

            plt.plot(epochs, loss, 'r', label='Training loss')  # ‘bo’는 파란색 점을 의미합니다.
            plt.plot(epochs, val_loss, 'b', label='Validation loss')  # ‘b’는 파란색 실선을 의미합니다.
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()
        print(result_list,"\navg result",sum(result_list)/10)
        return x_feats, y_labels


if "__name__"== "__main__":
    TM = TIMNET_Model()
    TM.test()

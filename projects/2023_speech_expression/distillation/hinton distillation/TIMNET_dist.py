"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import Conv1D, SpatialDropout1D,add,GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import sigmoid
from torchsummary import summary
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.utils import plot_model

def Temporal_Aware_Block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, name=''):
    #shape : (None, 188, 39) 동일
    original_x = x
    #1.1
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')(x)
    #print("1",conv_1_1.shape)
    conv_1_1 = BatchNormalization(trainable=True,axis=-1)(conv_1_1)
    #print("2",conv_1_1.shape)
    conv_1_1 =  Activation(activation)(conv_1_1)
    #print("3",conv_1_1.shape)
    output_1_1 =  SpatialDropout1D(dropout_rate)(conv_1_1)
    #print("4",output_1_1.shape)
    # 2.1
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')(output_1_1)
    #print("5",conv_2_1.shape)
    conv_2_1 = BatchNormalization(trainable=True,axis=-1)(conv_2_1)
    #print("6",conv_2_1.shape)
    conv_2_1 = Activation(activation)(conv_2_1)
    #print("7",conv_2_1.shape)
    output_2_1 =  SpatialDropout1D(dropout_rate)(conv_2_1)
    #print("8",output_2_1.shape)

    if original_x.shape[-1] != output_2_1.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
        
    output_2_1 = Lambda(sigmoid)(output_2_1)
    F_x = Lambda(lambda x: tf.multiply(x[0], x[1]))([original_x, output_2_1])
    return F_x


class TIMNET:
    def __init__(self,
                 nb_filters=128,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation = "relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='TIMNET'):
        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value=0.
        print("!!!!","nb_filters",self.nb_filters, "kernel_size",self.kernel_size, "dilation",self.dilations)

        if not isinstance(nb_filters, int):
            raise Exception()

    def __call__(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 11
        print("@@@@@@",self.dilations)
        forward = inputs
        backward = K.reverse(inputs,axes=1)
        
        #print("Input Shape=",inputs.shape)
        forward_convd = Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')(forward)
        backward_convd = Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')(backward)
        
        final_skip_connection = []
        
        skip_out_forward = forward_convd
        skip_out_backward = backward_convd

        #print("!!",self.nb_stacks, self.dilations) #1,8
        for s in range(self.nb_stacks): #0
            print("s",s)
            for i in [2 ** i for i in range(self.dilations)]: #1, 2, 4, 8, 16, 32, 64, 128
                print("i",i)
                skip_out_forward = Temporal_Aware_Block(skip_out_forward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size, 
                                                        self.dropout_rate,  
                                                        name=self.name)
                skip_out_backward = Temporal_Aware_Block(skip_out_backward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size, 
                                                        self.dropout_rate,  
                                                        name=self.name)
                
                temp_skip = add([skip_out_forward, skip_out_backward],name = "biadd_"+str(i))
                temp_skip=GlobalAveragePooling1D()(temp_skip)
                temp_skip=tf.expand_dims(temp_skip, axis=1)
                final_skip_connection.append(temp_skip)

        output_2 = final_skip_connection[0]
        #print("9",output_2.shape)
        for i,item in enumerate(final_skip_connection):
            if i==0:
                continue
            #print("!!",output_2.shape, item.shape)
            output_2 = K.concatenate([output_2,item],axis=-2)
            #print("@@@",output_2.shape)
        x = output_2  #(None, 8, 39)
        #print("final",x.shape)

        return x

def test():
    inputs = Input(shape=(172, 29))
    model = TIMNET()(inputs)
    plot_model(model, to_file='model_shapes.png', show_shapes=True)

#test()
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import tensorflow as tf
import numpy as np

import keras.backend as K

from keras.models import Model

SIZE = 224

from keras import layers


#contitional
q=False
# if q==True: #we have quantization ==> float16
#         import keras.backend as K
#         K.set_floatx('float16')
#         from bn import BatchNormalizationF16

def conv(x, channels, kernel, stride, name, use_bias=True, padding='SAME'):
    x = layers.Conv2D(filters=channels, kernel_size=kernel, strides=stride, use_bias=use_bias, padding=padding, name=name)(x)
    print(x.get_shape(), name, K.dtype(x))
    return x

def batch_norm(x, name):
    if (q==False): # float(32)
        return layers.BatchNormalization(epsilon=1e-05, center=True, scale=True)(x)
    else: #quatized version (float 16)
        return BatchNormalizationF16(name = name)(x)
def relu(x, name):
    x = layers.Activation('relu', name=name)(x)
    print(x.get_shape(), name, K.dtype(x))
    return x

def concat(arrs, axis, name):
    x= layers.concatenate(arrs, axis, name=name) #3 is the channel 
    print(x.get_shape(), name, K.dtype(x))
    return x
  

def maxpool2d(x, kernel, stride, name, padding='SAME'):
    # x = tf.nn.max_pool2d(x, kernel, stride, padding)
    x = layers.MaxPooling2D( (kernel,kernel), (stride,stride), padding)(x)
    print(x.get_shape(), name, K.dtype(x))
    return x

def avgpool2d(x, kernel, stride,name, padding='SAME'):
    x = layers.GlobalAveragePooling2D()(x)
    print(x.get_shape(), name, K.dtype(x))
    return x

def fc(x, units, name, use_bias=True):
    x = layers.Dense(units, activation='softmax', name=name)(x)    
    print(x.get_shape(), name, K.dtype(x))
    return x

#########################define network
def my_net_32():
    img_input = layers.Input(shape=[SIZE, SIZE, 3])

    Convolution1 = conv(img_input, 32, 3, 2, 'Conv2D', False,  padding='VALID')

    Scale1 = batch_norm(Convolution1, "BatchNorm1" )
    Scale1 = relu(Scale1, 'relu1')

    Convolution2 = conv(Scale1, 64, 3, 1, 'Conv2D_1', False,  padding='SAME')
    Scale2 = batch_norm(Convolution2,'BatchNorm2' ) #BN+Scale
    Scale2 = relu(Scale2, 'relu2')

    Pooling1 = maxpool2d(Scale2, 2, 2, 'pool1')#pool1 (x, kernel, stride, padding='SAME'):

    Convolution3 = conv(Pooling1, 64, 3, 1, 'Conv2D_2', False,  padding='SAME') #Conv2D_2
    Scale3 = batch_norm(Convolution3, 'BatchNorm3' ) #BN+Scale
    Scale3 = relu(Scale3, 'm1_b1_relu1')

    Convolution4 = conv(Pooling1, 48, 1, 1, 'Conv2D_3',  False,  padding='VALID') #Conv2D_3
    Scale4 = batch_norm(Convolution4, "BatchNorm4" ) #BN+Scale
    Scale4 = relu(Scale4, 'm1_b2_relu1')

    Scale4 = layers.ZeroPadding2D((2,2))(Scale4)
    Convolution5 = conv(Scale4, 64, 5, 1, 'Conv2D_4', False,  padding='VALID') #Conv2D_4 ###pad=2!!! 
    Scale5 = batch_norm(Convolution5, 'BatchNorm5' ) #BN+Scale
    Scale5 = relu(Scale5, 'm1_b2_relu2') 

    Eltwise1 = layers.add([Scale3,Scale5], name='Eltwise1')
    concat1 = concat([Eltwise1, Pooling1], axis=3, name='concat1')

    Convolution6 = conv(concat1, 96, 3, 2,'Conv2D_5', False,  padding='VALID') #Conv2D_5
    Scale6 = batch_norm(Convolution6, 'BatchNorm6' ) #BN+Scale
    Scale6 = relu(Scale6, 'm2_b1_relu1')

    Convolution7 = conv(concat1, 64, 1, 1, 'Conv2D_6', False,  padding='VALID') #Conv2D_6
    Scale7 = batch_norm(Convolution7, 'BatchNorm7' ) #BN+Scale
    Scale7 = relu(Scale7, 'm2_b2_relu1')

    Convolution8 = conv(Scale7, 96, 3, 1, 'Conv2D_7',False,  padding='SAME') #Conv2D_7
    Scale8 = batch_norm(Convolution8, 'BatchNorm8' ) #BN+Scale
    Scale8 = relu(Scale8, 'm2_b2_relu2')

    Convolution9 = conv(Scale8, 96, 3, 2, 'Conv2D_8', False,  padding='VALID') #Conv2D_8
    Scale9 = batch_norm(Convolution9, 'BatchNorm9' ) #BN+Scale
    Scale9 = relu(Scale9, 'm2_b2_relu2_1')

    Pooling2 = maxpool2d(concat1 , 4, 2, 'm2_b3_pool1', 'VALID')#m2_b3_pool1
    Pooling3 = maxpool2d(Pooling1, 4, 2, 'm2_b4_pool2', 'VALID')#m2_b4_pool2


    concat2 = concat([Pooling3, Scale6, Scale9, Pooling2], 3, 'concat2')  #??concat dim??

    Convolution10 = conv(concat2, 64, 3, 2, 'Conv2D_9', False,  padding='VALID') #Conv2D_9
    Scale10 = batch_norm(Convolution10, 'BatchNorm10' ) #BN+Scale
    Scale10 = relu(Scale10, 'relu6') #relu6

    Pooling4 = avgpool2d(Scale10, 4, 2, 'pool3')#pool3

    BiasAdd = fc(Pooling4,2, 'MatMul')

    model = Model(img_input, BiasAdd)

    #check the datatype
    # for l in model.layers:
    #     if ("Conv2D" in l.name):
    #        print(l.name, "type is: ", K.dtype(l.kernel))

    return model

# x = my_net_32()
# x=my_net()
# x.summary()
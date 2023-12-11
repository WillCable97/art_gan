import tensorflow as tf
import keras
from keras.layers import Conv2D, Activation, Conv2DTranspose, Concatenate, Input, BatchNormalization
from keras.models import Sequential, Model


"""*******************************GENERATOR*******************************"""
def DownSampleBlock(filters:int, kernel_size:int, strides:int):
    ret_block = Sequential()
    ret_block.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same'))
    ret_block.add(BatchNormalization(axis=-1))
    #ret_block.add(InstanceNormalization(axis=-1))
    ret_block.add(Activation('relu'))
    return ret_block

def UpSampleBlock(filters:int, kernel_size:int, strides:int):
    ret_block = Sequential()
    ret_block.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same'))
    ret_block.add(BatchNormalization(axis=-1))
    #ret_block.add(InstanceNormalization(axis=-1))
    ret_block.add(Activation('relu'))
    return ret_block
  
def Generator(filter_line:list[int], input_shape:tuple, kernel_size:int=3, strides:int=1) -> keras.Model:
    networks_appending = []
    input_block = Input(shape=input_shape)
    ret_block = input_block

    #Create encoder chain
    for filter_choice in filter_line:
        ret_block=DownSampleBlock(filter_choice,3,2)(ret_block)
        networks_appending.append(ret_block)

    #Decoder with residual connections
    for filter_choice, network_append in zip(filter_line[::-1][1:], networks_appending[::-1][1:]):
        ret_block = UpSampleBlock(filter_choice, 3,2)(ret_block)
        ret_block = Concatenate()([ret_block, network_append])
    
    ret_block = Conv2DTranspose(input_shape[-1], kernel_size, strides, padding='same', activation='tanh')(ret_block)

    return Model(input_block, ret_block)












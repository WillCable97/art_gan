import tensorflow as tf
import keras
from keras.layers import Conv2D, LeakyReLU, Input, BatchNormalization
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
import warnings

weight_initializer = RandomNormal(stddev=0.02)

"""*******************************DESCRIMINATOR*******************************"""
def DescriminatorBlock(filters:int, kernel_size:int, strides:int) -> tf.Tensor:
    ret_block =  Sequential()
    ret_block.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=weight_initializer))
    ret_block.add(BatchNormalization(axis=-1))
    ret_block.add(LeakyReLU(0.2))
    return ret_block


def Descriminator(input_shape:tuple, kernel_size:int=3, strides:int=1) -> keras.Model:
    discriminator_input = Input(shape=input_shape)
    ret_block = discriminator_input
    ret_block = DescriminatorBlock(64, kernel_size, strides)(ret_block)
    ret_block = DescriminatorBlock(128, kernel_size, strides)(ret_block)
    ret_block = DescriminatorBlock(256, kernel_size, strides)(ret_block)
    ret_block = DescriminatorBlock(512, kernel_size, strides)(ret_block)
    return Model(discriminator_input, ret_block)


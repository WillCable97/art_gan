import tensorflow as tf
from Processors import Processor

"""Instance of the processor class, resizes and normalises images for input, and recreates jpeg formatting for output"""

class GanImageProcessor(Processor):
    def preprocess_function(self, input_feature):
        img_data = tf.keras.utils.load_img(input_feature)
        return tf.reshape(tf.cast(tf.image.resize(img_data, (int(256), int(256))), tf.float32) / 127.5 - 1, (1, 256, 256, 3))
        
    def postprocess_function(self, output_feature):
        rescaled_img=(output_feature +  1) * 127.5
        return(rescaled_img.numpy()[0].astype('uint8'))
    
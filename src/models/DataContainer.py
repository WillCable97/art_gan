import os
import tensorflow as tf

"""
DATA CLASS FOR TENSOR FLOW RECORDS DATA
"""


class TensorflowDataObject:
    def __init__(self, filepath, feature_desc, data_col_name):
        self.feature_desc = feature_desc
        self.data_col_name = data_col_name
        self.tfrecord_file = [os.path.join(filepath, filename) for filename in os.listdir(filepath)]
        self.raw_dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        self.parsed_dataset = self.raw_dataset.map(self.parse_tfrecord_fn)
        self.batched_data = None
        self.current_batch = 0
        self.record_count = len(self.tfrecord_file)
    
    def parse_tfrecord_fn(self, record):
        """Function for parsing each record data, will require an input feature descrption"""
        feature_description=self.feature_desc
        example = tf.io.parse_single_example(record, feature_description)
        return example
    
    def return_single_eg(self): #Should be cleaned
        """Returns one sings example (the first one) from the data set"""
        for record in self.parsed_dataset:
            text_feature = record[self.data_col_name].numpy().decode('utf-8')
            return text_feature

    def create_batched_data(self, batch_size, buffer_size):
        """Creates batches in the data"""
        self.batched_data=self.parsed_dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size)

    def goto_next_batch(self):
        """Takes the next batch of data and increments the current batch variable"""
        next_batch_obj = self.batched_data.skip(self.current_batch).take(1)
        next_batch_dict = next(next_batch_obj.as_numpy_iterator())
        self.current_batch += 1
        return [element.decode('utf-8') for element in next_batch_dict[self.data_col_name]]
    
    def reset_batch(self):
        """Resets batch back to zero (usefull when training multuple epochs)"""
        self.current_batch = 0
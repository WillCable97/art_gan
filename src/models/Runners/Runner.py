from DataHandling.DataContainer import TensorflowDataObject
from Processors.Processor import Processor

import typing
import Errors.ErrorObject
import tensorflow as tf
from enum import Enum

from keras.optimizers import Optimizer, Adam
from keras.models import Model
from keras.losses import BinaryCrossentropy

"""Runner class responsible for controling data and training"""

class Runner:
    def __init__(self
                 , data_input : typing.Dict[str, TensorflowDataObject] #Names of datasets (domains)
                 , pipline_processor : Processor #Processor (input / ouptut cleaning)
                 , model_collection : typing.Dict[str, Model] #Names of models
                 , batch_count : int #Number of batches in training data
                 , latest_epoch: int #The modst recent epoch
                 , hyper_params: Enum
                 ):
        #Function Inputs
        self.data_input = data_input
        self.pipline_processor = pipline_processor
        self.model_collection = model_collection
        self.batch_count = batch_count
        self.model_objnames = [model_name for model_name in model_collection]
        self.current_epoch = latest_epoch
        self.model_hyperparams = hyper_params

        #Details for training run
        self.error_tracking = Errors.ErrorObject.ErrorHandler(self.model_objnames)
        self.input_optimizers = {key_val: Adam(learning_rate=0.0002, beta_1=0.5) for key_val in self.model_collection}
        self.entropy_loss = BinaryCrossentropy(from_logits=True) 


    #Handling batch and epoch runs
    def run_epoch(self, saving=True):
        self.current_epoch += 1

        for i in range(self.batch_count):
            print(f"Running batch: {i}")
            self.run_batch()
            break
        

    #Handle the batch
    def run_batch(self): #Need to clean up this
        refined_data = []

        #Run batch through pre-precessor
        for training_set in self.data_input:
            next_batch_data = self.data_input[training_set].goto_next_batch()
            processed_batch_data = list(map(self.pipline_processor.preprocess_function, next_batch_data))
            refined_data.append(processed_batch_data)
        
        
        #Loop over data and perform run step
        zipped_data = zip(*refined_data)
        for k, zipped_data_instance in enumerate(zipped_data):
            errors = self.run_step(zipped_data_instance)
            self.error_tracking.read_in_vals(errors, self.current_epoch)
            break
        
    def run_step(self, inputs):
        """Implementation dependant on model configuration"""



            

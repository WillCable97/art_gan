from DataContainer import TensorflowDataObject
import Models
import Processors

import typing
import ErrorObject
from enum import Enum
import os
import math

from keras.optimizers import Optimizer, Adam
from keras.models import Model
from keras.losses import BinaryCrossentropy

"""Runner class responsible for controling data and training"""

class Runner:
    def __init__(self
                 , data_input : typing.Dict[str, typing.Dict[str, TensorflowDataObject]] #Names of datasets (domains)
                 
                 , model_hyperparams : Enum #Random collection of parameters
                 , pipline_processor : Processors.Processor #Processor (input / ouptut cleaning)
                 , model_collection : typing.Dict[str, Model] = None #Names of models
                 #, load_from_file = False #Will load from file when doing epochs seperatly
                 ):
        #Save instance variables
        self.data_input = data_input
        self.model_hyperparams = model_hyperparams
        self.pipline_processor = pipline_processor
        self.model_collection = model_collection

        self._run_env_setup(self.model_hyperparams.ModelPath.value, self.model_hyperparams.ModelName.value) #Create environment



    def _run_env_setup(self, model_path:str, model_name:str):
        pass


        """
        self.input_optimizers = {key_val: Adam(learning_rate=0.0002, beta_1=0.5) for key_val in model_collection}
        self.train_data = self.data_input['train']
        self.batch_size = model_hyperparams.BatchSize.value
        self.model_name = model_hyperparams.ModelName.value

        self.entropy_loss = BinaryCrossentropy(from_logits=True) #This shouldn't be in the base class 
        self.epoch_number = 1 if not load_from_file else self.load_model_data() #1
        self.error_tracking = ErrorObject.ErrorHandler([key for key in self.model_collection])
        self.batch_count = self._get_batch_count()
        
        #Batch the training data
        for i, key_val in enumerate(self.train_data): self.train_data[key_val].create_batched_data(self.batch_size, 1000)
        """

    def run_epoch(self, saving=False):
        for i in range(self.batch_count):
            print(f"Running batch: {i}")
            self.run_batch()
            #break
            
        if saving: self.run_model_save()

    def run_batch(self):
        refined_data = []

        for training_set in self.train_data:
            next_batch_data = self.train_data[training_set].goto_next_batch()
            processed_batch_data = list(map(self.pipline_processor.preprocess_function, next_batch_data))
            refined_data.append(processed_batch_data)
        
        zipped_data = zip(*refined_data)

        for k, zipped_data_instance in enumerate(zipped_data):
            errors = self.run_step(zipped_data_instance)
            self.error_tracking.read_in_vals(errors)
            #break

    def run_step(self, inputs):
        """Implementation dependant on model configuration"""
         

    def run_model_save(self):
        for (model, f_path) in self._get_model_paths(self.epoch_number): 
            if not os.path.exists(f_path): os.makedirs(f_path)

            self.model_collection[model].save_weights(os.path.join(f_path, model))

        self.error_tracking.save_json_obj(os.path.join(self._get_epoch_path(self.epoch_number), 'error_metrics.json'))


    def load_model_data(self, use_most_recent = True, epoch_num = 1):
        last_epoch = epoch_num if not use_most_recent else self._get_last_epoch()

        for (model, f_path) in self._get_model_paths(last_epoch): 
            self.model_collection[model].load_weights(os.path.join(f_path, model))

        return last_epoch + 1



    def _get_model_paths(self, input_epoch) -> list[str]:
        epoch_folder = self._get_epoch_path(input_epoch)
        return [(model, os.path.join(epoch_folder, model)) for model in self.model_collection] 

    def _get_last_epoch(self): #Dont like this, should be cleaner way
        model_path = self.model_hyperparams.ModelPath.value
        epoch_folder = os.path.join(model_path, self.model_name)
        max_epoch = 1

        for folder_name in os.listdir(epoch_folder):
            found_epoch_num = int(str(folder_name).replace("Epoch",""))
            if found_epoch_num > max_epoch : max_epoch = found_epoch_num
        
        return max_epoch

    def _get_epoch_path(self, input_epoch):
        model_path = self.model_hyperparams.ModelPath.value 
        return os.path.join(model_path, self.model_name, f"Epoch{input_epoch}") 

    def _get_batch_count(self) -> int :
        #Do batch count
        dataset_size = self.train_data[next(iter(self.train_data))].record_count
        return math.ceil(dataset_size / self.batch_size)

            

from DataContainer import TensorflowDataObject
import Models
import Processors
import EnvObj

import typing
import ErrorObject
from enum import Enum
import os
import math
import tensorflow as tf

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
                 ):
        #Function Inputs
        self.data_input = data_input
        self.model_hyperparams = model_hyperparams
        self.pipline_processor = pipline_processor
        self.model_collection = model_collection
        
        #Some variables for simplification 
        self.train_data = self.data_input['train']
        self.batch_size = self.model_hyperparams.BatchSize.value
        self.model_name = self.model_hyperparams.ModelName.value
        self.model_base_path = self.model_hyperparams.ModelPath.value

        #Environment
        self.env_obj = EnvObj.TrainingEnvironmentObject(self.model_base_path, self.model_name)
        self._handle_model_init()
        self.epoch_number = self.env_obj._get_epoch_number()
        self.model_path = self.env_obj.project_path

        #Other
        self.batch_count = self._get_batch_count()
        self.error_tracking = ErrorObject.ErrorHandler([key for key in self.model_collection])
        self.input_optimizers = {key_val: Adam(learning_rate=0.0002, beta_1=0.5) for key_val in self.model_collection}
        self.entropy_loss = BinaryCrossentropy(from_logits=True) 

        #Batch training data 
        for i, key_val in enumerate(self.train_data): self.train_data[key_val].create_batched_data(self.batch_size, 1000)
        
    #Handling model files
    def _handle_model_init(self):
        models_init = self.env_obj.models_init

        if models_init: 
            pass
            #Load the models
            #models_to_load = self.env_obj.model_directories_for_load()
            #self.model_collection = {model: tf.keras.models.load_model(models_to_load[model]) for model in models_to_load}

        else: 
            #Save the models
            models_to_save = self.env_obj.model_directories_for_save(self.model_collection)
            for model in models_to_save:
                self.model_collection[model].save(models_to_save[model]) 

    def run_model_weight_saves(self):
        models_to_save = self.env_obj.directories_for_weight_save(self.epoch_number, self.model_collection)
        for model in models_to_save:
            self.model_collection[model].save_weights(models_to_save[model])
        self.env_obj._create_param_json(self.model_hyperparams, self.epoch_number)

    def run_model_weight_load(self):
        models_to_load = self.env_obj.directories_for_weight_load()
        for model in self.model_collection: self.model_collection[model].load_weights(models_to_load[model])   


    def run_epochs(self, epoch_count : int, saving=False) -> None:
        for i in range(epoch_count): 
            print(f"Running Epoch {i}")
            self.run_epoch(saving) 

    #Handling batch and epoch runs
    def run_epoch(self, saving=False):
        self.epoch_number = self.env_obj._get_epoch_number() + 1
        for i in range(self.batch_count):
            print(f"Running batch: {i}")
            self.run_batch()
            #break
        
        if not saving: return #no more action required
        
        self.run_model_weight_saves()
        self.env_obj._increment_epoch()

    def run_batch(self): #Need to clean up this
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

    def _get_batch_count(self) -> int :
        #Do batch count
        dataset_size = self.train_data[next(iter(self.train_data))].record_count
        return math.ceil(dataset_size / self.batch_size)

            

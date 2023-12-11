"""
TEMPLATE TO CHANGE:
    PROCESSOR
    RUNNER
    DATA
    MODELS
"""


##################PACKAGESS##################
#Public imports
import sys
import os
from enum import Enum
import tensorflow as tf
import typing
from keras.models import Model
import math

#Local imports
from  Processors.GanImageProcessor import GanImageProcessor as ProcessorUsed #Change to base class again
from Runners.GanRunner import CycleGanRunner as RunnerUsed 
from Models.GanV2 import DescriminatorV2
from Models.GanV2 import GeneratorV2

#Base imports
import DataHandling.DataContainer as DataContainer
from Environement.EnvObj import EnvObject


##################MODEL PARAMETERS##################
class hyper_params(Enum):
    BatchSize = 32 
    LAMBDA=10 
    ModelName='GanV2'


##################ENVIRONMENT####################
parent_dir  = os.path.abspath('.')
env_object = EnvObject(os.path.abspath('.'), hyper_params.ModelName.value)
raw_data_folder = os.path.join(parent_dir, 'data', 'raw')
processed_data_folder = os.path.join(parent_dir, 'data', 'processed')


##################MODELS##################

filter_line = [64,128,256,512,512,512,512,512]
model_set:typing.Dict[str, Model] = {        
        'a_b_generator' : GeneratorV2.Generator(filter_line, (256,256,3), 3,2),
        'b_a_generator' : GeneratorV2.Generator(filter_line, (256,256,3), 3,2),
        'a_descrim' : DescriminatorV2.Descriminator((256,256,3),4,2),
        'b_descrim' : DescriminatorV2.Descriminator((256,256,3),4,2)}


##################MODEL ENVIRONMENT##################
env_object.model_object_env_init(model_set) #need to add in model save here
latest_epoch = env_object.read_epoch_number()

#Load weights if this isnt the first run 
if not latest_epoch == 0 : 
    paths_to_weights = env_object.training_weights_for_epoch(latest_epoch)
    for model_name in model_set: model_set[model_name].load_weights(paths_to_weights[model_name])
    print(f"Loaded weights from epoch {latest_epoch}")


##################FEATURE MAPS##################
dat_col_name = 'text_feature' #Update
feature_description = {dat_col_name: tf.io.FixedLenFeature([], dtype=tf.string),}


##################DATA##################
training_list = ['trainA', 'trainB'] #Update
records_paths = [os.path.join(os.path.join(processed_data_folder, f_name)) for f_name in training_list]
domain_containers = [DataContainer.TensorflowDataObject(records_path, feature_description, dat_col_name) for records_path in records_paths]

main_container = {'train': {key: val for key, val in zip(records_paths, domain_containers)}}


##################TRAINING DATA##################
training_data = main_container['train']
#Batch data
for i, key_val in enumerate(training_data): training_data[key_val].create_batched_data(hyper_params.BatchSize.value, 1000)
dataset_size = training_data[next(iter(training_data))].record_count
batch_count = math.ceil(dataset_size / hyper_params.BatchSize.value)

pipline_proc = ProcessorUsed()


##################RUNNER##################
runner = RunnerUsed(training_data, pipline_proc, model_set, batch_count, latest_epoch, hyper_params)


##################TRAINING LOOP##################
for i in range(1):
    print(f"Running epoch{runner.current_epoch + 1}")
    runner.run_epoch()
    env_object.epoch_init(runner.current_epoch, hyper_params)
    paths_to_weights = env_object.training_weights_for_epoch(runner.current_epoch)
    for model_name in runner.model_collection: runner.model_collection[model_name].save_weights(paths_to_weights[model_name])
    env_object.change_json_epoch(runner.current_epoch)
    env_object.add_errors(runner.current_epoch, runner.error_tracking.full_error_object.get(runner.current_epoch, {}))










##################PACKAGESS##################
#Public imports
import sys
import os
from enum import Enum
import tensorflow as tf
import typing
from keras.models import Model

#Local imports
#import Processors.GanImageProcessor as ProcessorUsed
#import Runners.GanRunner as RunnerUsed
from Models.GanV1 import DescriminatorV1
from Models.GanV1 import GeneratorV1

#Base imports
import DataHandling.DataContainer as DataContainer
from Environement.EnvObj import EnvObject


##################MODEL PARAMETERS##################
class hyper_params(Enum):
    BatchSize = 32 
    LAMBDA=10 
    ModelName='GanV1Test'


##################ENVIRONMENT####################
parent_dir  = os.path.abspath('.')
env_object = EnvObject(os.path.abspath('.'), hyper_params.ModelName.value)
raw_data_folder = os.path.join(parent_dir, 'data', 'raw')
processed_data_folder = os.path.join(parent_dir, 'data', 'processed')


##################MODELS##################
filter_line = [64,128,256,512,512] #For simplicity for now
model_set:typing.Dict[str, Model] = {        
        'a_b_generator' : GeneratorV1.Generator(filter_line, (256,256,3), 3,2),
        'b_a_generator' : GeneratorV1.Generator(filter_line, (256,256,3), 3,2),
        'a_descrim' : DescriminatorV1.Descriminator((256,256,3),3,2),
        'b_descrim' : DescriminatorV1.Descriminator((256,256,3),3,2)}


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

#pipline_proc = ProcessorUsed.Processor()




##################RUNNER##################
#runner = RunnerUsed.Runner(main_container, hyper_params, pipline_proc, model_set)
#runner.run_epoch(True)
























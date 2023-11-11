import os 
import DataContainer
import Models
from GanImageProcessor import GanImageProcessor
import GanRunner
from enum import Enum
import tensorflow as tf
import Evaluator
import EnvObj



#File paths
base_folder=os.path.abspath('./')
raw_data_folder= os.path.join(base_folder, 'data', 'raw')
processed_data_folder=os.path.join(base_folder, 'data', 'processed')
records_path=os.path.join(os.path.join(processed_data_folder, 'trainA'))
records_path2=os.path.join(os.path.join(processed_data_folder, 'trainB'))
path_to_models=os.path.join(base_folder, 'models')






#Feature map
feature_description = {'text_feature': tf.io.FixedLenFeature([], dtype=tf.string),}

#Data figures
domainA_contianer = DataContainer.TensorflowDataObject(records_path, feature_description, 'text_feature')
domainB_container = DataContainer.TensorflowDataObject(records_path2, feature_description, 'text_feature')
main_container = {
    'train': {
        'traina' : domainA_contianer,
        'trainb' : domainB_container
    }
}

#Models
filter_line = [64,128,256,512,512] #For simplicity for now
model_set = {
        'a_b_generator' : Models.Generator(filter_line, (256,256,3), 3,2),
        'b_a_generator' : Models.Generator(filter_line, (256,256,3), 3,2),
        'a_descrim' : Models.Descriminator((256,256,3),3,2),
        'b_descrim' : Models.Descriminator((256,256,3),3,2)
}

class hyper_params(Enum):
    BatchSize = 32 
    LAMBDA=10 
    ModelName='MyTest'
    ModelPath=path_to_models


pipline_proc = GanImageProcessor()


#A = Runner.Runner(main_container, model_set, hyper_params, pipline_proc)

A = GanRunner.CycleGanRunner(main_container, hyper_params, pipline_proc, model_set)
A.run_epoch(True)


"""
model_path = os.path.join(path_to_models, 'Image Gan V2')
B = Evaluator.Evaluator(model_path)
B.add_runner(GanRunner.CycleGanRunner(main_container, model_set, hyper_params, pipline_proc))
B.visualise_outputs('b_a_generator')
#B.graph_error_models()

"""





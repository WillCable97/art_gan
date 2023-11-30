import os 
import json
import matplotlib.pyplot as plt
import Runner
import Models
import EnvObj

class Evaluator:
    def __init__(self, base_path, project_name, all_epoch = True):
        self.base_path = base_path
        self.project_name = project_name
        self.env_obj = EnvObj.TrainingEnvironmentObject(self.base_path, self.project_name)
        self.model_schema = self.load_model_names()
        self.full_error_obj = {model_name: [] for model_name in self.model_schema}
        self.runner_obj: Runner.Runner = None
        self.loop_epochs()
        #print(self.full_error_obj)

    def load_model_names(self):
        models_folder = os.path.join(self.env_obj.project_path, "models")
        return [model_name for model_name in os.listdir(models_folder)]

    def add_runner(self, input_runner):
        self.runner_obj = input_runner

    def loop_epochs(self):
        epoch_list = self.create_epoch_list()
        for epoch in epoch_list: 
            self.add_epoch_errors(epoch)

    def create_epoch_list(self):
        current_epoch = self.env_obj._get_epoch_number()
        return range(1, current_epoch+1)
    

    def add_epoch_errors(self, epoch_number):
        path_to_json = self.env_obj._path_to_error_object(epoch_number)

        with open(path_to_json, 'r') as json_file:
            raw_data = json.load(json_file)

        for key in self.full_error_obj: self.full_error_obj[key] += raw_data[key]
        
    def graph_error_models(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for k, model_name in enumerate(self.full_error_obj):
            column = k % 2
            row = int(k / 2)
            axs[row,column].plot(list(map(float, self.full_error_obj[model_name])))
            axs[row,column].set_title(model_name)
        
        plt.tight_layout()
        plt.show()
        
    def visualise_outputs(self, input_model):
        epoch_list = self.create_epoch_list()
        raw_example = self.runner_obj.train_data['trainb'].return_single_eg()
        proc_example = self.runner_obj.pipline_processor.preprocess_function(raw_example)
        list_of_outputs = [proc_example]
        
        for epoch in epoch_list:
            self.runner_obj.run_model_weight_load(epoch)
            model_to_use = self.runner_obj.model_collection[input_model]
            next_output = model_to_use(proc_example)
            list_of_outputs.append(next_output)
        
        output_list = list(map(self.runner_obj.pipline_processor.postprocess_function, list_of_outputs))

        fig, axs = plt.subplots(1, len(epoch_list)+1, figsize=(10, 8))

        for k, image in enumerate(output_list):
            axs[k].imshow(image)

        plt.show()

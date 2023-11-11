import os 
import json
import matplotlib.pyplot as plt
import Runner
import Models


class Evaluator:
    def __init__(self, model_path, all_epoch = True):
        self.model_path = model_path
        self.full_error_obj = {}
        self.model_schema = []
        self.init_done = False
        self.runner_obj: Runner.Runner = None

        self.loop_epochs()

    def add_epoch_errors(self, epoch_number):
        full_file_path = os.path.join(self.model_path, f"Epoch{epoch_number}", "error_metrics.json")
        with open(full_file_path, 'r') as json_file:
            raw_data = json.load(json_file)

        loaded_data = {k : [float(i) for i in v] for k,v in raw_data.items()}

        if self.init_done: 
            for key in self.model_schema:
                self.full_error_obj[key] += loaded_data[key]
        else:
            self.full_error_obj = loaded_data
            self.model_schema = [key for key in loaded_data]
            self.init_done = True

    def loop_epochs(self):
        epoch_list = self.create_epoch_list()
        for epoch in epoch_list: self.add_epoch_errors(epoch)

    def create_epoch_list(self):
        ret_list = []

        for fold_name in os.listdir(self.model_path):
            epoch_nummber = str(fold_name).replace("Epoch", "")
            ret_list.append(int(epoch_nummber))
            a=[]
            
        return ret_list#.sort()
    
    def add_runner(self, input_runner):
        self.runner_obj = input_runner
    
    def graph_error_models(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for k, model_name in enumerate(self.full_error_obj):
            column = k % 2
            row = int(k / 2)
            axs[row,column].plot(self.full_error_obj[model_name])
            axs[row,column].set_title(model_name)
        
        plt.tight_layout()

        plt.show()
        
    def visualise_outputs(self, input_model):
        epoch_list = self.create_epoch_list()
        raw_example = self.runner_obj.train_data['trainb'].return_single_eg()
        print(raw_example)
        proc_example = self.runner_obj.pipline_processor.preprocess_function(raw_example)
        list_of_outputs = [proc_example]
        
        for epoch in epoch_list:
            self.runner_obj.load_model_data(False, epoch)
            model_to_use = self.runner_obj.model_collection[input_model]
            next_output = model_to_use(proc_example)
            list_of_outputs.append(next_output)
        
        output_list = list(map(self.runner_obj.pipline_processor.postprocess_function, list_of_outputs))

        fig, axs = plt.subplots(1, len(epoch_list)+1, figsize=(10, 8))

        for k, image in enumerate(output_list):
            axs[k].imshow(image)

        plt.show()



        


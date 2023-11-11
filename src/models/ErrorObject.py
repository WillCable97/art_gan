import json

"""Object for tracking and saving error data during training"""

class ErrorHandler:
    def __init__(self, model_name_schema):
        self.run_output = []
        self.model_name_schema = model_name_schema
        self.json_obj = {}
 
    def read_in_vals(self, input_vals):
        """Adds new set of error values (after each run of gradient descent)"""
        self.run_output.append(input_vals)
    
    def create_json_obj(self):
        """Returns json object for all errors added to the run output list"""
        for k, model_name in enumerate(self.model_name_schema):
            self.json_obj[model_name] = [str(i[k]) for i in self.run_output]
        return self.json_obj
    
    def save_json_obj(self, file_path):
        """Creates a json object and saves it to specified file path"""
        self.create_json_obj()
        json_obj = json.dumps(self.json_obj)
        with open(file_path, 'w') as json_file:
            json_file.write(json_obj)


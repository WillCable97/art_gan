import json

"""Object for tracking and saving error data during training"""

class ErrorHandler:
    def __init__(self, model_objnames: list):
        self.full_error_object = {}
        self.model_objnames = model_objnames

    def read_in_vals(self, input_vals: list, current_epoch:int):
        if not current_epoch in self.full_error_object: 
            self.full_error_object[current_epoch] = {model_name: [] for model_name in self.model_objnames} #Init Epoch

        for k, model_name in enumerate(self.model_objnames):
            self.full_error_object[current_epoch][model_name].append(list(map(float, input_vals)))
    
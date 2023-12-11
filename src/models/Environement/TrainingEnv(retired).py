import os
import json

class TrainingEnvironmentObject:
    def __init__(self, base_path, project_name):
        self.project_path = os.path.join(base_path, project_name)
        self._check_project_existence()
        self.models_init = self._models_are_init()

    #Model functions
    def model_directories_for_save(self, input_model_obj):
        self._create_fodler(os.path.join(self.project_path, "models"))
        return {model: self._path_to_model_save(model) for model in input_model_obj}

    def model_directories_for_load(self):
        models_location = os.path.join(self.project_path, "models")
        return {model: self._path_to_model_save(model) for model in os.listdir(models_location)}
    
    def directories_for_weight_save(self, epoch_num, input_model_obj):
        self._create_fodler(self._path_to_epoch(epoch_num))
        return {model: self._path_to_model_in_epoch(epoch_num, model) for model in input_model_obj}

    def directories_for_weight_load(self, epoch_num):
        models_location = self._path_to_epoch(epoch_num)
        return {model: self._path_to_model_in_epoch(epoch_num, model) for model in os.listdir(models_location)}

    #Environment init
    def _check_project_existence(self):
        already_exists = self._create_fodler(self.project_path)
        if not already_exists: self._create_project_details()

    def _create_project_details(self):
        self._create_json_file(self._path_to_project_detail(), {"epoch": 0})

    def _create_param_json(self, input_enum, epoch):
        json_obj = {member.name: member.value for member in input_enum}
        self._create_json_file(self._path_to_params_in_epoch(epoch), json_obj)

    

    #Reading and querey
    def _read_project_details_object(self):
        json_path = self._path_to_project_detail()
        with open(json_path) as json_file:
            parsed_json = json.load(json_file)
        return parsed_json

    def _models_are_init(self) -> bool:
        return self._check_folder_existence(os.path.join(self.project_path, "models"))
    
    def _get_epoch_number(self):
        return self._read_project_details_object()['epoch']
    
    def _increment_epoch(self):
        json_file = self._read_project_details_object()
        json_file["epoch"] += 1
        self._create_json_file(self._path_to_project_detail(), json_file)


    #General functions
    def _create_json_file(self, file_path, content):
        with open(file_path, 'w') as json_file:
            json_file.write(json.dumps(content))

    def _check_folder_existence(self, folder_path):
        return os.path.exists(folder_path)
    
    def _create_fodler(self, folder_path:str) -> bool:
        folder_exists = self._check_folder_existence(folder_path)
        if not folder_exists: os.makedirs(folder_path)
        return folder_exists
    
    #File Paths
    def _path_to_project_detail(self) -> str:
        return os.path.join(self.project_path, "project_details.json")

    def _path_to_model_save(self, model:str):
        return os.path.join(self.project_path, "models", model)

    def _path_to_epoch(self, epoch_number:int):
        return os.path.join(self.project_path, f"Epoch{epoch_number}")

    def _path_to_model_in_epoch(self, epoch_number:int, model:str):
        epoch_folder = self._path_to_epoch(epoch_number)
        return os.path.join(epoch_folder, model, model)
    
    def _path_to_params_in_epoch(self, epoch_number:int):
        return os.path.join(self._path_to_epoch(epoch_number), "run_params.json")
    
    def _path_to_error_object(self, epoch_number:int):
        return os.path.join(self._path_to_epoch(epoch_number), "error_metrics.json")























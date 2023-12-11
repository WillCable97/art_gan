import os
import json
from Environement.HelperFuncs.file_ops import contruct_file_path, check_folder_existence, create_fodler, create_json_file, read_json_file
from enum import Enum

class EnvObject:
    def __init__(self, project_path:str, model_name:str):
        self.project_name = os.path.basename(project_path)
        self.project_path = project_path
        self.model_path = os.path.join(project_path, 'models', model_name)
        self.project_details = os.path.join(self.model_path, 'project_details.json')
        self.error_file = os.path.join(self.model_path, 'errors.json')
        self.model_objects = []
        self.paths_of_model_objects = []

        #Initialize environment
        self.model_env_init()

    def model_env_init(self) -> None:
        #Check environemnt detail existtence
        env_exists = check_folder_existence(self.model_path)
        pro_details_exists = check_folder_existence(self.project_details)
        errors_exists = check_folder_existence(self.error_file)

        #Create environemnt params
        if not env_exists: create_fodler(self.model_path)
        if not pro_details_exists: create_json_file(self.project_details, {"epoch": 0})
        if not errors_exists: create_json_file(self.error_file, {})

    def model_object_env_init(self, model_set: dict) -> None: #more than just dict
        object_parent = os.path.join(self.model_path, 'models')

        for model_name in model_set:
            model_obj_path = os.path.join(object_parent, model_name)
            model_obj_exists = check_folder_existence(model_obj_path)
            if not model_obj_exists: create_fodler(model_obj_path)

            #Add to instance variables
            self.model_objects.append(model_name)
            self.paths_of_model_objects = model_obj_path

    def read_epoch_number(self) -> int:
        contents = read_json_file(self.project_details)
        return contents['epoch']
    

    def epoch_folder(self, epoch_number:int):
        return os.path.join(self.model_path, f"Epoch{epoch_number}")
    
    def training_weights_for_epoch(self, epoch:int) -> list: #will return dict with name and file path
        return {model_name: os.path.join(self.epoch_folder(epoch), model_name, model_name) for model_name in self.model_objects}
    
    def epoch_init(self, epoch_number: int, train_params: Enum):
        epoch_folder = self.epoch_folder(epoch_number)
        #Check environemnt detail existence
        epoch_exist = check_folder_existence(epoch_folder)

        if epoch_exist: return 

        create_fodler(epoch_folder)
        create_json_file(os.path.join(epoch_folder, 'run_params.json'), {attr.name: attr.value for attr in train_params})

        for model_name in self.model_objects:
            create_fodler(os.path.join(epoch_folder, model_name))

    def change_json_epoch(self, epoch:int):
        contents = read_json_file(self.project_details)
        contents['epoch'] = epoch
        create_json_file(self.project_details, contents)
        
    def add_errors(self, epoch:int, content:dict):
        contents = read_json_file(self.error_file)
        contents[epoch] = content
        create_json_file(self.error_file, contents)






    
    


    
    







    
    






    
"""
    def model_env_init(self):
        #Check if the folder already exists
        fodler_exists = self._check_folder_existence(self.model_path)
        if fodler_exists: return 

        #Create the folder and add the project details
        self._create_fodler(self.model_path)
        

        







    def _create_env(self, model_path:str):




        self._create_fodler()


    def _check_project_existence(self):
        already_exists = self._create_fodler(self.project_path)
        if not already_exists: self._create_project_details()

    def _create_project_details(self):
        self._create_json_file(self._path_to_project_detail(), {"epoch": 0})


"""









"""







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

"""





















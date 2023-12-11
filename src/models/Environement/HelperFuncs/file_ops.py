import os
import json

"""General file operations"""

def create_fodler(folder_path:str):
    folder_exists = check_folder_existence(folder_path)
    if not folder_exists: os.makedirs(folder_path)

def check_folder_existence(folder_path) -> bool:
    return os.path.exists(folder_path)

def contruct_file_path(base_path, *argv):
    ret_path = base_path

    for path in argv:
        ret_path = os.path.join(ret_path, path)
    return ret_path


def create_json_file(file_path:str, content:dict):
    with open(file_path, 'w') as json_file:
        json_file.write(json.dumps(content))

def read_json_file(file_path: str):
    with open(file_path) as json_file:
        parsed_json = json.load(json_file)
    return parsed_json


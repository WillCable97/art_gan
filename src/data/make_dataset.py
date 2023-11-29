# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import os
import tensorflow as tf
import numpy as np 






@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    base_folder=os.path.abspath('./')
    raw_data_folder= os.path.join(base_folder, 'data', 'raw')
    processed_data_folder=os.path.join(base_folder, 'data', 'processed')
    sub_dir_list=['testA', 'testB', 'trainA', 'trainB']

    for sub_dir in sub_dir_list:
        img_paths, img_name = create_list_of_img_paths(img_dir=os.path.join(raw_data_folder, sub_dir))
        
        for [file_path, file_name] in zip(img_paths, img_name):
            save_tf_records(file_path, os.path.join(processed_data_folder, sub_dir) ,file_name )






def create_list_of_img_paths(img_dir: str):
    return_list = [] 
    file_names = []
    counter=0
    print (img_dir)

    for filename in os.listdir(img_dir):
        return_list.append(os.path.join(img_dir, filename))
        file_names.append(str.replace(filename, '.jpg', ''))

        counter+=1
        if counter > 1072 :
            break #only one for now
    
    return return_list, file_names



# Define a function to create a TFExample from your data
def create_tf_example(data):
    feature = {
        'text_feature': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data['text_feature'].encode('utf-8')])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example



def save_tf_records(image_file_path : str, tf_record_base_path : str, file_name : str): 
    # Define the data you want to save
    data_to_save = {
        'text_feature': image_file_path,  # Modify with your string data
    }

    if not os.path.exists(tf_record_base_path): os.makedirs(tf_record_base_path)

    
    # Specify the path to save the TFRecord
    tfrecord_file = os.path.join(tf_record_base_path, file_name + '.tfrecord')

    # Create a TFRecordWriter and write the example
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        example = create_tf_example(data_to_save)
        writer.write(example.SerializeToString())




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()






    
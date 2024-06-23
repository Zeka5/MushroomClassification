import sys
import numpy as np
import sklearn as sk
import os
import cv2
from sklearn.model_selection import train_test_split

def load_images(image_paths):
    dataset = {}

    for path in image_paths:
        path_parts = path.split("\\")
        mushroom_type = path_parts[4]
        image_name = path_parts[5]

        key = image_name + "," + mushroom_type
        image_path = "data\\" + mushroom_type + "\\" + image_name
        value = cv2.imread(image_path)

        dataset[key] = value

    return dataset

def load_data(root_dir):
    agaricus = []
    amanita = []
    boletus = []
    cortinarius = []
    entoloma = []
    hygrocybe = []
    lactarius = []
    russula = []
    suillus = []

    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Full path to the image
            image_path = os.path.join(dirname, filename)
            
            # Determine mushroom type based on directory name
            mushroom_type = os.path.basename(dirname)
            
            # Append the image path to the corresponding mushroom type list
            if mushroom_type == "Agaricus":
                agaricus.append(image_path)
            elif mushroom_type == "Amanita":
                amanita.append(image_path)
            elif mushroom_type == "Boletus":
                boletus.append(image_path)
            elif mushroom_type == "Cortinarius":
                cortinarius.append(image_path)
            elif mushroom_type == "Entoloma":
                entoloma.append(image_path)
            elif mushroom_type == "Hygrocybe":
                hygrocybe.append(image_path)
            elif mushroom_type == "Lactarius":
                lactarius.append(image_path)
            elif mushroom_type == "Russula":
                russula.append(image_path)
            elif mushroom_type == "Suillus":
                suillus.append(image_path)
            else:
                pass

    train_agaricus, temp_agaricus = train_test_split(agaricus, test_size=0.2, random_state=42)
    val_agaricus, test_agaricus = train_test_split(temp_agaricus, test_size=0.5, random_state=42)

    train_amanita, temp_amanita = train_test_split(amanita, test_size=0.2, random_state=42)
    val_amanita, test_amanita = train_test_split(temp_amanita, test_size=0.5, random_state=42)

    train_boletus, temp_boletus = train_test_split(boletus, test_size=0.2, random_state=42)
    val_boletus, test_boletus = train_test_split(temp_boletus, test_size=0.5, random_state=42)

    train_cortinarius, temp_cortinarius = train_test_split(cortinarius, test_size=0.2, random_state=42)
    val_cortinarius, test_cortinarius = train_test_split(temp_cortinarius, test_size=0.5, random_state=42)

    train_entoloma, temp_entoloma = train_test_split(entoloma, test_size=0.2, random_state=42)
    val_entoloma, test_entoloma = train_test_split(temp_entoloma, test_size=0.5, random_state=42)

    train_hygrocybe, temp_hygrocybe = train_test_split(hygrocybe, test_size=0.2, random_state=42)
    val_hygrocybe, test_hygrocybe = train_test_split(temp_hygrocybe, test_size=0.5, random_state=42)

    train_lactarius, temp_lactarius = train_test_split(lactarius, test_size=0.2, random_state=42)
    val_lactarius, test_lactarius = train_test_split(temp_lactarius, test_size=0.5, random_state=42)

    train_russula, temp_russula = train_test_split(russula, test_size=0.2, random_state=42)
    val_russula, test_russula = train_test_split(temp_russula, test_size=0.5, random_state=42)

    train_suillus, temp_suillus = train_test_split(suillus, test_size=0.2, random_state=42)
    val_suillus, test_suillus = train_test_split(temp_suillus, test_size=0.5, random_state=42)

    train_set = (
        train_agaricus + train_amanita + train_boletus + train_cortinarius + 
        train_entoloma + train_hygrocybe + train_lactarius + train_russula + train_suillus
    )
    val_set = (
        val_agaricus + val_amanita + val_boletus + val_cortinarius + 
        val_entoloma + val_hygrocybe + val_lactarius + val_russula + val_suillus
    )
    test_set = (
        test_agaricus + test_amanita + test_boletus + test_cortinarius + 
        test_entoloma + test_hygrocybe + test_lactarius + test_russula + test_suillus
    )

    # Return the datasets
    return train_set, val_set, test_set

if __name__=="__main__":
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    os.chdir(parent_directory)
    new_directory = os.getcwd()
    data_dir = os.path.join(new_directory, 'data')

    training_images, validation_images, test_images = load_data(data_dir)
    print("Loading training dataset...")
    training_dataset = load_images(training_images)
    print("Training dataset loaded")
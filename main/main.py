import sys
import numpy as np
import sklearn as sk
import os
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from skimage.feature import hog
import random

def load_images(image_paths, augment=False):
    dataset = {}
    target_size=(64, 64)

    for path in image_paths:
        path_parts = path.split("\\")
        mushroom_type = path_parts[4]
        image_name = path_parts[5]

        key = image_name + "," + mushroom_type
        image_path = "data\\" + mushroom_type + "\\" + image_name
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize image to target size
        image = cv2.resize(image, target_size)

        if augment:
            flipped_image = cv2.flip(image, 1)  # Flip horizontally
            flipped_key = "flipped_" + key
            dataset[flipped_key] = flipped_image

        dataset[key] = image

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

def extract_features_and_labels(dataset):
    features = []
    labels = []
    
    for key, image in dataset.items():
        image_name, label = key.split(",")
        hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features = hog_features.flatten()
        features.append(hog_features)
        labels.append(label)
        
    return np.array(features), np.array(labels)

def train_and_evaluate(x_treniranje, y_treniranje, x_validacija, y_validacija):
    print("Classifier training...")

    param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100], 'kernel': ['poly']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_treniranje, y_treniranje)

    best_classifier = grid_search.best_estimator_
    y_train_pred = best_classifier.predict(x_treniranje)
    y_test_pred = best_classifier.predict(x_validacija)

    treniranje_accuracy = accuracy_score(y_treniranje, y_train_pred)
    validacija_accuracy = accuracy_score(y_validacija, y_test_pred)

    print("Best parameters found: ", grid_search.best_params_)
    print("Training accuracy: ", treniranje_accuracy)
    print("Validation accuracy: ", validacija_accuracy)
    print()

    return best_classifier


def test_classifier(classifier, test_images):
    print("Testing...")
    test_dataset = load_images(test_images, False)  # No augmentation for test images"

    x_test, y_test = extract_features_and_labels(test_dataset)

    y_test_pred = classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy: ", test_accuracy * 100, "%")
    print()
    return test_accuracy


if __name__=="__main__":
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    os.chdir(parent_directory)
    new_directory = os.getcwd()
    data_dir = os.path.join(new_directory, 'data')

    training_images, validation_images, test_images = load_data(data_dir)
    print("Loading training dataset...")
    training_dataset = load_images(training_images, True)
    print("Training dataset loaded, loaded " + str(len(training_dataset)) + " images")
    print("Loading validation dataset...")
    validation_dataset = load_images(validation_images, False)  #no need to augment validation dataset
    print("Validation dataset loaded, loaded " + str(len(validation_dataset)) + " images")
    print()

    #whole dataset
    #x_training, y_training = extract_features_and_labels(training_dataset)
    #x_validation, y_validation = extract_features_and_labels(validation_dataset)
    #best_classifier = train_and_evaluate(x_training, y_training, x_validation, y_validation)
    #test_accuracy = test_classifier(best_classifier, test_images)

    #subset for faster computing
    subset_size = len(training_dataset) // 10
    training_subset_keys = random.sample(list(training_dataset.keys()), subset_size)
    training_subset = {key: training_dataset[key] for key in training_subset_keys}
    subset_size = len(validation_dataset) // 10
    validation_subset_keys = random.sample(list(validation_dataset.keys()), subset_size)
    validation_subset = {key: validation_dataset[key] for key in validation_subset_keys}
    x_training, y_training = extract_features_and_labels(training_subset)
    x_validation, y_validation = extract_features_and_labels(validation_subset)
    best_classifier = train_and_evaluate(x_training, y_training, x_validation, y_validation)
    test_accuracy = test_classifier(best_classifier, test_images)

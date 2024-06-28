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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input

import certifi
import ssl

def load_images(image_paths, augment=False):
    dataset = {}
    target_size=(64, 64)

    for path in image_paths:
        path_parts = path.split("\\")
        mushroom_type = path_parts[-2]
        image_name = path_parts[-1]

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

def load_images_for_cnn(image_paths, augment=False):
    dataset = []
    labels = []
    target_size = (64, 64)

    for path in image_paths:
        path_parts = path.split("\\")
        mushroom_type = path_parts[-2]
        image_name = path_parts[-1]

        image_path = os.path.join("data", mushroom_type, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, target_size)
        image = image / 255.0  # normalize pixels to [0, 1]

        dataset.append(image)
        labels.append(mushroom_type)

        if augment:
            flipped_image = cv2.flip(image, 1)  # Flip horizontally
            dataset.append(flipped_image)
            labels.append(mushroom_type)

    return np.array(dataset), np.array(labels)

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

def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Use Input layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
    param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100], 'kernel': ['poly']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_treniranje, y_treniranje)

    best_classifier = grid_search.best_estimator_
    y_train_pred = best_classifier.predict(x_treniranje)
    y_test_pred = best_classifier.predict(x_validacija)

    training_accuracy = accuracy_score(y_treniranje, y_train_pred)
    validation_accuracy = accuracy_score(y_validacija, y_test_pred)

    print("Training accuracy: ", training_accuracy)
    print("Validation accuracy: ", validation_accuracy)
    print()

    return best_classifier

def train_cnn_model(x_train, y_train, x_val, y_val, num_classes):
    input_shape = x_train.shape[1:]
    model = create_cnn_model(input_shape, num_classes)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32
    )

    return model

def evaluate_cnn_model(model, x_test, y_test, num_classes):
    y_test = to_categorical(y_test, num_classes)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print()
    print("CNN test accuracy: ", test_accuracy * 100, "%")
    print()
    return test_accuracy


def test_classifier(classifier, test_images):
    test_dataset = load_images(test_images, False)  # No augmentation for test images"

    x_test, y_test = extract_features_and_labels(test_dataset)

    y_test_pred = classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy: ", test_accuracy * 100, "%")
    print()
    return test_accuracy

def cnn_subset_data(x_data, y_data, subset_fraction=0.1):
    subset_size = int(len(x_data) * subset_fraction)
    indices = np.random.choice(len(x_data), subset_size, replace=False)
    x_subset = x_data[indices]
    y_subset = y_data[indices]
    return x_subset, y_subset

def build_resnet_model(input_shape, num_classes):
    ssl._create_default_https_context = ssl._create_unverified_context
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_resnet_model(model, x_train, y_train, x_val, y_val, epochs=10):
    try:
        print("Starting initial training with frozen base model layers...")
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(x_val, y_val)
        )

        print("\nInitial training completed. Unfreezing the last 10 layers and fine-tuning...")
        for layer in model.layers[-10:]:
            layer.trainable = True

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(x_val, y_val)
        )

        return model
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def evaluate_resnet_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return test_accuracy

if __name__=="__main__":
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    os.chdir(parent_directory)
    new_directory = os.getcwd()
    data_dir = os.path.join(new_directory, 'data')
    training_images, validation_images, test_images = load_data(data_dir)

    print("\n\nLoading and preprocessing datasets for CNN...")
    x_train, y_train = load_images_for_cnn(training_images, augment=True)
    x_val, y_val = load_images_for_cnn(validation_images, augment=False)
    x_test, y_test = load_images_for_cnn(test_images, augment=False)

    #neural networks require numeric input, this maps labels to numbers
    input_shape = (64, 64, 3)
    num_classes = len(np.unique(y_train))
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    # whith whole dataset
    # print("Training CNN model...")
    # cnn_model = train_cnn_model(x_train, y_train, x_val, y_val, input_shape, num_classes)


    # subsetting the training and validation data for CNN
    subset_fraction = 0.1  # 10% of data for fair comparison witn svm
    x_train_subset, y_train_subset = cnn_subset_data(x_train, y_train, subset_fraction)
    x_val_subset, y_val_subset = cnn_subset_data(x_val, y_val, subset_fraction)
    num_classes = len(le.classes_)
    print("Training CNN model on subset...")
    cnn_model = train_cnn_model(x_train_subset, y_train_subset, x_val_subset, y_val_subset, num_classes)

    print("Evaluating CNN model...")
    cnn_test_accuracy = evaluate_cnn_model(cnn_model, x_test, y_test, num_classes)

    y_train_subset = tf.keras.utils.to_categorical(y_train_subset, num_classes=num_classes)
    y_val_subset = tf.keras.utils.to_categorical(y_val_subset, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    print("\nBuilding and training ResNet model...\n")
    resnet_model = build_resnet_model(input_shape, num_classes)
    resnet_model = train_resnet_model(resnet_model, x_train_subset, y_train_subset, x_val_subset, y_val_subset)

    print("\nEvaluating ResNet model...")
    resnet_test_accuracy = evaluate_resnet_model(resnet_model, x_test, y_test)

    # comparing with SVM
    print("\nLoading datasets for SVM...")
    training_dataset = load_images(training_images, True)
    validation_dataset = load_images(validation_images, False)

    #whole dataset
    #x_training, y_training = extract_features_and_labels(training_dataset)
    #x_validation, y_validation = extract_features_and_labels(validation_dataset)
    #best_classifier = train_and_evaluate(x_training, y_training, x_validation, y_validation)
    #test_accuracy = test_classifier(best_classifier, test_images)

    #subset (10%) for faster computing
    subset_size = len(training_dataset) // 10
    training_subset_keys = random.sample(list(training_dataset.keys()), subset_size)
    training_subset = {key: training_dataset[key] for key in training_subset_keys}
    subset_size = len(validation_dataset) // 10
    validation_subset_keys = random.sample(list(validation_dataset.keys()), subset_size)
    validation_subset = {key: validation_dataset[key] for key in validation_subset_keys}
    
    x_training, y_training = extract_features_and_labels(training_subset)
    x_validation, y_validation = extract_features_and_labels(validation_subset)

    print("\nTraining SVM model...")
    best_classifier = train_and_evaluate(x_training, y_training, x_validation, y_validation)

    print("Evaluating SVM model...")
    svm_test_accuracy = test_classifier(best_classifier, test_images)

    print("\n\n-----------------------------------------------------------------------------------------")
    print("COMPARISON OF TEST ACCURACIES")
    print("-----------------------------------------------------------------------------------------")
    print(f"CNN Test Accuracy: {cnn_test_accuracy * 100:.2f}%")
    print(f"ResNet Test Accuracy: {resnet_test_accuracy * 100:.2f}%")
    print(f"SVM Test Accuracy: {svm_test_accuracy * 100:.2f}%")
    print("\n\n")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import optuna
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 150
LABELS = ['PNEUMONIA', 'NORMAL']

def get_training_data(data_dir):
    """Load and preprocess training data."""
    data = []
    for label in LABELS:
        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    continue
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return np.array(data, dtype=object)

def create_model(trial):
    """Create a CNN model with hyperparameters to be optimized."""
    # Hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 2, 5)
    initial_filters = trial.suggest_int('initial_filters', 16, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    model = keras.Sequential()
    
    # First layer
    model.add(layers.Conv2D(initial_filters, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Additional layers
    for i in range(n_layers - 1):
        filters = initial_filters * (2 ** (i + 1))
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model, batch_size

def objective(trial):
    """Objective function for Optuna optimization."""
    # Load data
    train_data = get_training_data(os.path.join('data', 'chest_xray', 'train'))
    val_data = get_training_data(os.path.join('data', 'chest_xray', 'val'))
    
    # Prepare data for training
    X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array([i[1] for i in train_data])
    X_val = np.array([i[0] for i in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_val = np.array([i[1] for i in val_data])
    
    # Normalize pixel values
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    # Create model
    model, batch_size = create_model(trial)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=trial.suggest_int('rotation_range', 0, 30),
        width_shift_range=trial.suggest_float('width_shift_range', 0, 0.2),
        height_shift_range=trial.suggest_float('height_shift_range', 0, 0.2),
        horizontal_flip=trial.suggest_categorical('horizontal_flip', [True, False]),
        fill_mode='nearest'
    )
    
    # Training
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[
            TFKerasPruningCallback(trial, 'val_accuracy'),
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
        ]
    )
    
    return history.history['val_accuracy'][-1]

def main():
    """Main function to run the hyperparameter optimization."""
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join('results', 'optimization_history.png'))
    plt.close()
    
    # Plot parameter importances
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join('results', 'parameter_importances.png'))
    plt.close()
    
    # Save best model
    best_model, _ = create_model(trial)
    best_model.save(os.path.join('results', 'best_model'))

if __name__ == "__main__":
    main() 
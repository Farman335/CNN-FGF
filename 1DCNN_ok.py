import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Define the 1D CNN model
def create_model(input_size, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(input_size, 1), padding='same'))  # Apply "same" padding
    model.add(MaxPooling1D(pool_size=2, padding='same'))  # Apply "same" padding
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))  # Apply "same" padding
    model.add(MaxPooling1D(pool_size=2, padding='same'))  # Apply "same" padding
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))  # Apply "same" padding
    model.add(MaxPooling1D(pool_size=2, padding='same'))  # Apply "same" padding
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

# Define cross-validation function
def cross_validate(X, y, num_folds, num_epochs, batch_size, learning_rate):
    fold_size = len(X) // num_folds
    metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'auc': [], 'mcc': []}
    models = []  # Store the trained models

    kf = KFold(n_splits=num_folds, shuffle=True)

    for train_indices, val_indices in kf.split(X):
        train_X, val_X = X[train_indices], X[val_indices]
        train_y, val_y = y[train_indices], y[val_indices]

        # Reshape input data for 1D CNN
        train_X = np.expand_dims(train_X, axis=2)
        val_X = np.expand_dims(val_X, axis=2)

        # Create model
        num_classes = y.shape[1]
        model = create_model(train_X.shape[1], num_classes)

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train model
        model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, verbose=0)

        # Evaluate model
        val_predictions = model.predict(val_X)
        val_predictions_binary = np.round(val_predictions).flatten()

        # Compute evaluation metrics
        accuracy = accuracy_score(np.argmax(val_y, axis=1), np.argmax(val_predictions, axis=1))
        tn, fp, fn, tp = confusion_matrix(np.argmax(val_y, axis=1), np.argmax(val_predictions, axis=1)).ravel()

        # Handle division by zero for sensitivity and specificity
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        # Handle division by zero for MCC
        mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt(mcc_denominator) if mcc_denominator != 0 else 0.0

        # Add metrics to dictionary
        metrics['accuracy'].append(accuracy)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        metrics['auc'].append(roc_auc_score(val_y[:, 1], val_predictions[:, 1]))
        metrics['mcc'].append(mcc)

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(val_y[:, 1], val_predictions[:, 1])
        plt.plot(fpr, tpr, label='Fold')

        # Store the trained model
        models.append(model)

    # Plot averaged ROC curve
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()

    return metrics, models  # Return the trained models as well

# Example usage
iRec = 'DDE_Steroid_receptor_train_1001_1029.csv'
D = pd.read_csv(iRec, header=None)

# Split into features (X) and classes (y)
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values
y = to_categorical(y)  # Convert labels to one-hot encoding

num_folds = 5
num_epochs = 50
batch_size = 100
learning_rate = 0.001

metrics, models = cross_validate(X, y, num_folds, num_epochs, batch_size, learning_rate)
avg_accuracy = np.mean(metrics['accuracy'])
avg_sensitivity = np.mean(metrics['sensitivity'])
avg_specificity = np.mean(metrics['specificity'])
avg_auc = np.mean(metrics['auc'])
avg_mcc = np.mean(metrics['mcc'])

# Save the trained model

# Save the model to a JSON file
model_json = models[0].to_json()  # Use the first model from the list
with open("1DCNN_model_1.json", "w") as json_file:
    json_file.write(model_json)
models[0].save_weights("1DCNN_model_1.h5")
print("Saved model to disk")

print(f'Average Accuracy: {avg_accuracy}')
print(f'Average Sensitivity: {avg_sensitivity}')
print(f'Average Specificity: {avg_specificity}')
print(f'Average AUC: {avg_auc}')
print(f'Average MCC: {avg_mcc}')

# Deep Learning Exercise

* To run the exercise, execute all the code blocks in the attached Jupyter notebook.
* All test cases will run automatically—no manual parameter changes are needed.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Loading Data](#loading-data)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
- [Analysis](#analysis)
- [Reference](#reference)

## Installation

To run this project, you need Python 3.7 and the following packages installed:

```bash
!python --version
!sudo apt-get install python3.7
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
pip install tensorflow numpy matplotlib
```

## Usage
### Loading Data

```python
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, metrics, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.callbacks import TensorBoard, Callback

# Load FashionMNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
label_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Augment images with horizontal flips for specific classes
def augment_images_with_flips(images, labels, target_classes, label_names):
    target_indices = [label_names.index(cls) for cls in target_classes]
    augmented_images = []
    augmented_labels = []
    for img, lbl in zip(images, labels):
        augmented_images.append(img)
        augmented_labels.append(lbl)
        if lbl in target_indices:
            flipped_img = np.fliplr(img)
            augmented_images.append(flipped_img)
            augmented_labels.append(lbl)
    return np.array(augmented_images), np.array(augmented_labels)

target_classes = ['Sandal', 'Sneaker', 'Bag', 'Ankle boot']
augmented_images, augmented_labels = augment_images_with_flips(x_train, y_train, target_classes, label_names)
x_train, y_train = augmented_images, augmented_labels

# Normalize data
x_train = normalize(x_train)
x_test = normalize(x_test)

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

### Training the Model
```python
# Define the LeNet-5 model with optional regularization techniques
def create_model(optimizer, loss, metrics, learning_rate, Batch_norm, Dropout, dropout_rate, L2, seed):
    if L2:
        optimizer = optimizers.AdamW()
    optimizer.learning_rate.assign(learning_rate)
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializers.HeUniform(seed), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    if Dropout:
        model.add(layers.Dropout(rate=dropout_rate[0]))
    if Batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer=initializers.HeUniform(seed)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    if Dropout:
        model.add(layers.Dropout(rate=dropout_rate[1]))
    if Batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(120, activation='relu', kernel_initializer=initializers.HeUniform(seed)))
    if Batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(84, activation='relu', kernel_initializer=initializers.HeUniform(seed)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model

# Callback to track test metrics
class TestMetrics(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.test_losses = []
        self.test_accuracies = []
    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)
        logs['test_loss'] = test_loss
        logs['test_accuracy'] = test_accuracy

# Train and save the model
def train_and_save_model(batch_norm, l2, dropout, model_name, x_train, y_train):
    optimizer = optimizers.Adam()
    loss = 'categorical_crossentropy'
    metrics = 'accuracy'
    validation_split = 0.2
    learning_rate = 5e-4
    epochs = 10
    batch_size = 128
    dropout_rate = [0.15, 0.5]
    seed = 152

    model = create_model(optimizer, loss, metrics, learning_rate, batch_norm, dropout, dropout_rate, l2, seed)
    log_dir = f"/content/drive/MyDrive/ex1_332450014_906943097/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    TB_callback = [TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch='5, 10', embeddings_freq=1)]
    test_data_callback = TestMetrics(test_data=(x_test, y_test))

    history = model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[TB_callback, test_data_callback])
    model.save(f"/content/drive/MyDrive/ex1_332450014_906943097/{model_name}.h5")

    return history, test_data_callback.test_losses, test_data_callback.test_accuracies

# Train models with different regularization techniques
history_baseline, test_losses_baseline, test_accuracies_baseline = train_and_save_model(batch_norm=False, l2=False, dropout=False, model_name="Baseline", x_train=x_train, y_train=y_train)
history_batch_norm, test_losses_batch_norm, test_accuracies_batch_norm = train_and_save_model(batch_norm=True, l2=False, dropout=False, model_name="Batch_norm", x_train=x_train, y_train=y_train)
history_dropout, test_losses_dropout, test_accuracies_dropout = train_and_save_model(batch_norm=False, l2=False, dropout=True, model_name="Dropout", x_train=x_train, y_train=y_train)
history_l2, test_losses_l2, test_accuracies_l2 = train_and_save_model(batch_norm=False, l2=True, dropout=False, model_name="L2", x_train=x_train, y_train=y_train)
```

### Testing the Model
```python
from tensorflow.keras import models

# Test the trained models
def test_models(x_test, y_test):
    results = {}
    model_names = ['Baseline', 'Batch_norm', 'Dropout', 'L2']
    for model_name in model_names:
        model = models.load_model(f"/content/drive/MyDrive/ex1_332450014_906943097/{model_name}.h5")
        test_loss, test_acc = model.evaluate(x_test, y_test)
        results[model_name] = [test_loss, test_acc]
    return results

results = test_models(x_test, y_test)

# Plot convergence graphs and final results
def plot_test_metrics(train_accuracy, val_accuracy, test_accuracies, model_name, test_results):
    epochs = range(1, len(test_accuracies) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_accuracy, 'bo-', label='Train accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
    plt.plot(epochs, test_accuracies, 'go-', label='Test accuracy')
    plt.title(f'{model_name} - Train/Test Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    test_results[model_name] = [train_accuracy[-1], test_accuracies[-1]]
    return test_results

test_results = {}
test_results = plot_test_metrics(history_baseline.history['accuracy'], history_baseline.history['val_accuracy'], test_accuracies_baseline, "Baseline", test_results)
test_results = plot_test_metrics(history_batch_norm.history['accuracy'], history_batch_norm.history['val_accuracy'], test_accuracies_batch_norm, "Batch_norm", test_results)
test_results = plot_test_metrics(history_dropout.history['accuracy'], history_dropout.history['val_accuracy'], test_accuracies_dropout, "Dropout", test_results)
test_results = plot_test_metrics(history_l2.history['accuracy'], history_l2.history['val_accuracy'], test_accuracies_l2, "L2", test_results)

model_names = ['Baseline', 'Batch_norm', 'Dropout', 'L2']
test_results = np.array([[model_name, test_results[model_name][0], test_results[model_name][1]] for model_name in model_names])

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=test_results, colLabels=[" ", "Train Accuracy", "Test Accuracy"], cellLoc='center', loc='center')
table.set_fontsize(12)
table.scale(1.2, 1.2)
ax.set_title('Model Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
plt.show()
```

## Analysis
* Best Regularization Technique: Batch normalization consistently outperforms other techniques with the highest test accuracy of 0.8919 and train accuracy of 0.9468. It effectively stabilizes the training process and reduces overfitting.
* Dropout and L2 Regularization: Both techniques provide improvements over the baseline in terms of test accuracy. Dropout helps in preventing overfitting with a slight trade-off in training accuracy, while L2 regularization provides a balanced improvement.
* Baseline vs Regularized Models: All regularization techniques improved the model's performance compared to the baseline. The test accuracy of the baseline model is lower, indicating more overfitting compared to the regularized models.

## Reference
[LeNet Architecture: A Complete Guide](https://www.kaggle.com/code/blurredmachine/lenet-architecture-a


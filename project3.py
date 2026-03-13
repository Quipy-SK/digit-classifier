"""
=======================================================
HANDWRITTEN DIGIT CLASSIFIER
=======================================================
Sanzhar Kali
Dataset: MNIST — 70,000 handwritten digit images
Goal: Recognize digits 0-9 from images
Model: Neural Network with ReLU and Softmax
 
What this project demonstrates:
- Neural network architecture (Dense layers)
- ReLU activation function
- Softmax for multiclass classification
- Adam optimizer
- Loss function (SparseCategoricalCrossentropy)
- Training loop with epochs
- Model evaluation
 
Learned concepts from Machine Learning ourse Andrew Ng

HOW TO RUN:
pip install tensorflow matplotlib numpy
python digit_classifier.py
=======================================================
"""
 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
 
print("Training images:", X_train.shape)
print("Test images:", X_test.shape)
 
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()
for i in range(16):
    random_index = np.random.randint(0, len(X_train))
    axes[i].imshow(X_train[random_index], cmap='gray')
    axes[i].set_title(f"Label: {y_train[random_index]}", fontsize=10)
    axes[i].axis('off')
plt.suptitle('Sample MNIST Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150)
plt.close()
 
X_train = X_train / 255.0
X_test  = X_test  / 255.0
 
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)
 
model = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')
])
 
model.summary()
 
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
 
history = model.fit(
    X_train_flat,
    y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
 
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)

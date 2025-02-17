import numpy as np
import keras
from matplotlib import pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Normalize the images.
train_images = (train_images / 255)
test_images = (test_images / 255)

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)


# Build the model with added convolutional layer and BatchNormalization.
model = keras.models.Sequential([
    keras.layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.2),  # Lower dropout rate to reduce over-regularization.
    keras.layers.Dense(16, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Define callbacks for dynamic learning rate adjustment and early stopping.
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train the model with callbacks.
history = model.fit(
    train_images,
    keras.utils.to_categorical(train_labels),
    epochs=20,
    validation_data=(test_images, keras.utils.to_categorical(test_labels)),
    callbacks=callbacks
)

loss, accuracy = model.evaluate(test_images, keras.utils.to_categorical(test_labels))

print('Test accuracy:', accuracy)

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('Model accuracy')  
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

import tensorflow as tf

# Assume 'model' is your already-trained Keras model.

# Convert the model to TFLite format with dynamic range quantization.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized model to a file.
with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized TFLite model saved as 'model_quantized.tflite'.")


MODEL_NAME = 'cnn'
model.save(f'models/{MODEL_NAME}.keras')


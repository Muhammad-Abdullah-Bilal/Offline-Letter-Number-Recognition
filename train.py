import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# 2. Load EMNIST Balanced dataset (Contains 0-9, A-Z, a-z)
# Total 47 balanced classes
print("Downloading dataset... this may take a minute.")
dataset, info = tfds.load('emnist/balanced', with_info=True, as_supervised=True)
train_ds, test_ds = dataset['train'], dataset['test']

# 3. Preprocess Data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST images are flipped and rotated by default; this fixes it
    image = tf.image.transpose(image) 
    return image, label

train_ds = train_ds.map(preprocess).cache().shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32)

# 4. Build the CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(47, activation='softmax') 
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train the model
print("Starting training...")
model.fit(train_ds, epochs=10, validation_data=test_ds)

# 6. Save the model
model.save('char_recognition_model.h5')
print("Finished! Download 'char_recognition_model.h5' from the folder icon on the left.")
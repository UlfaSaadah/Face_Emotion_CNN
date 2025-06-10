import tensorflow as tf
from tensorflow.keras import layers, models

# Contoh data dummy
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

# Model sederhana
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model dan menyimpan history
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val))

# Tampilkan akurasi terakhir
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"Akurasi training terakhir: {train_acc * 100:.2f}%")
print(f"Akurasi validasi terakhir: {val_acc * 100:.2f}%")

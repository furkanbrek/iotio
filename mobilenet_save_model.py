import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GPU configuration
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU memory growth'u etkinleştir
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Görünür cihazları ayarla
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU")

# CUDA optimization için mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load MNIST dataset
(images_data_train, images_train_labels), (images_data_test, images_test_labels) = tf.keras.datasets.mnist.load_data()

# Limit the dataset size
images_train_labels = images_train_labels[:1000]
images_test_labels = images_test_labels[:1000]
images_data_train = images_data_train[:1000]
images_data_test = images_data_test[:1000]

# Create data generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    validation_split=0.2
)

# Add channel dimension and normalize
images_data_train = np.expand_dims(images_data_train, axis=-1) / 255.0
images_data_test = np.expand_dims(images_data_test, axis=-1) / 255.0

train_generator = datagen.flow(images_data_train, images_train_labels, batch_size=32)
test_generator = datagen.flow(images_data_test, images_test_labels, batch_size=32)

# Görüntüleri 224x224 boyutuna yeniden boyutlandırma
def resize_images(images):
    resized_images = np.zeros((images.shape[0], 224, 224, 3))
    for i in range(images.shape[0]):
        # Expand to 3 channels and resize
        img = np.repeat(images[i], 3, axis=-1)
        resized_images[i] = cv2.resize(img, (224, 224))  # OpenCV ile yeniden boyutlandırma
    return resized_images

# Görüntüleri yeniden boyutlandır
images_data_train_resized = resize_images(images_data_train)
images_data_test_resized = resize_images(images_data_test)

def Make_model():
    My_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling='avg'
    )
    
    # Add custom classification head
    x = My_model.output
    predictions = tf.keras.layers.Dense(10, activation='softmax', dtype='float32')(x)
    
    # Create the full model
    My_model = tf.keras.Model(inputs=My_model.input, outputs=predictions)

    # CUDA optimization için optimizer ayarları
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    My_model.compile(optimizer=optimizer,
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return My_model

# Create a basic model instance
My_model = Make_model()

# Display the model's architecture
My_model.summary()

path_checkpoint = "training_1/cp.weights.h5"
directory_checkpoint = os.path.dirname(path_checkpoint)

# Ensure the checkpoint directory exists
os.makedirs(directory_checkpoint, exist_ok=True)

# CUDA optimization için daha iyi performans sağlayan ayarlar
callback = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=path_checkpoint,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
]

# Configure batch size for better GPU utilization
BATCH_SIZE = 64  # Increase batch size for GPU
BUFFER_SIZE = 1000

# Create tf.data.Dataset for better performance
train_dataset = tf.data.Dataset.from_tensor_slices((images_data_train_resized, images_train_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((images_data_test_resized, images_test_labels))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Train the model
My_model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    callbacks=callback
)

# Evaluate the model
loss, accuracy = My_model.evaluate(test_dataset, verbose=2)
print("Model accuracy: {:5.2f}%".format(100 * accuracy))

# Save the model
os.makedirs('saved_model', exist_ok=True)
My_model.save_weights('training_1/cp.weights.h5')
print("Ağırlıklar Kaydedildi")
My_model.save('saved_model/my_model.keras')

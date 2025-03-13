import os

import tensorflow as tf

from tensorflow import keras

(images_data_train, images_train_labels), (images_data_test, images_test_labels) = tf.keras.datasets.mnist.load_data()

images_train_labels = images_train_labels[:1000]

images_test_labels = images_test_labels[:1000]

images_data_train = images_data_train[:1000].reshape(-1, 28 * 28) / 255.0

images_data_test = images_data_test[:1000].reshape(-1, 28 * 28) / 255.0


def Make_model():

    My_model = tf.keras.models.Sequential([

        keras.layers.Dense(512, activation='relu', input_shape=(784,)),

        keras.layers.Dropout(0.2),

        keras.layers.Dense(10)

    ])

    My_model.compile(optimizer='adam',

                     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),

                     metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return My_model


# Create a basic model instance

My_model = Make_model()

# Display the model's architecture

My_model.summary()

path_checkpoint = "training_1/cp.weights.h5"

directory_checkpoint = os.path.dirname(path_checkpoint)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,

                                                 save_weights_only=True,

                                                 verbose=1)

My_model.fit(images_data_train,

          images_train_labels,

          epochs=10,

          validation_data=(images_data_test, images_test_labels),

          callbacks=[callback])

My_model = Make_model()

loss, accuracy_d = My_model.evaluate(images_data_test, images_test_labels, verbose=2)

print("Untrained model, accuracy: {:5.2f}%".format(100 * accuracy_d))

My_model.load_weights(path_checkpoint)

loss, accuracy_d = My_model.evaluate(images_data_test, images_test_labels, verbose=2)

print("Restored model, accuracy: {:5.2f}%".format(100 * accuracy_d))

os.makedirs('saved_model', exist_ok=True)

My_model.save_weights('training_1/cp.weights.h5')
print("Ağırlıklar Kaydedildi")
My_model.save('saved_model/my_model.keras')

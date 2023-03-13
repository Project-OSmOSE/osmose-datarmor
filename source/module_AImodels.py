

from scipy import ndimage

# build the image folder with a classname-based structure
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


path_osmose_home = "/home/datawork-osmose/"
path_osmose_dataset = "/home/datawork-osmose/dataset/"



def naivePSDbased(welch_values,fPSD):

    ww = welch_values[:, fPSD > 30]
    f = fPSD[fPSD > 30]

    perc = np.percentile(ww, 99, 0, interpolation='linear')
    sp_99 = ndimage.median_filter( perc , size=20)
    diff_sp = np.sum(abs(perc - sp_99))

    return diff_sp



def simpleCNNtensorflow(dataset_ID,task_ID,test_percent):

    # Algo simple CNN tensorflow see https://www.tensorflow.org/tutorials/images/transfer_learning 
    
    path_AI_dataset = os.path.join(path_osmose_dataset, dataset_ID,'analysis','AI','task_'+str(task_ID),'dataset')

    batch_size = 64
    img_height = 180
    img_width = 180

    train_dataset = tf.keras.utils.image_dataset_from_directory(
      path_AI_dataset,
      validation_split=2*test_percent/100,
      subset="training",
      shuffle=True,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
      path_AI_dataset,
      validation_split=test_percent/100,
      subset="validation",
      shuffle=True,    
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    class_names = train_dataset.class_names
#     print(class_names)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_dataset.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")


    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.Dropout(.2),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.Dropout(.2),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.Dropout(.2),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

#     model.summary()

    epochs=40
    history = model.fit(
      train_dataset,
      validation_data=validation_dataset,
      epochs=epochs,
      verbose=0
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

#     print('Predictions:\n', predictions.numpy())
#     print('Labels:\n', label_batch)

    # plt.figure(figsize=(10, 10))
    # for i in range(9):
    #   ax = plt.subplot(3, 3, i + 1)
    #   plt.imshow(image_batch[i].astype("uint8"))
    #   plt.title(class_names[predictions[i]])
    #   plt.axis("off")


#     img = tf.keras.utils.load_img(
#         list_images[0], target_size=(img_height, img_width)
#     )

#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0) # Create a batch

#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])

#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score))
#     )
    
    metadata_path = os.path.join(path_osmose_dataset, dataset_ID,'analysis','AI','task_'+str(task_ID),'simpleCNNtensorflow.h5')
    model.save(metadata_path)
             
             
    return accuracy, metadata_path




import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2  # the open cv library for python


# Download the dataset. Just a small sample to reduce training time
tfds.disable_progress_bar()
# 80% train, 10% validation and test
trainData = tfds.load('cats_vs_dogs', split='train[:5%]', as_supervised=True, shuffle_files=True)
validationData = tfds.load('cats_vs_dogs', split='train[5%:7%]', as_supervised=True, shuffle_files=True)
testData = tfds.load('cats_vs_dogs', split='train[7%:9%]', as_supervised=True, shuffle_files=True)


# A function for getting the image into the required format
def formatImage(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (240, 240))
    image = (image / 255)
    return image, label


# Batch the data
trainDataBatched = trainData.map(formatImage).batch(16)
validationDataBatched = validationData.map(formatImage).batch(16)
testDataBatched = testData.map(formatImage).batch(10)


# Define the CNN model
def buildModel():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(240, 240, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),  # this is required
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=RMSprop(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


# Build the model
model = buildModel()


history = model.fit(trainDataBatched, epochs=20, validation_data=validationDataBatched)


# A function to turn the image to the required format for input into the model
def preProcessImage(img):
    return np.expand_dims(img, axis=0)


# the output of the model summary will help you to get the name of the final convolution layer
model.summary()


# The function to generate CAM
def generateClassicalCAM(model, imgArray, predicted_class):
    # Get the weights of the final dense layer (before sigmoid)
    finalDenseLayerWeights = model.layers[-1].get_weights()[0]
    
    # Get the output of the last convolutional layer
    convOutput = model.get_layer("conv2d_4").output  # Put the name of the final convolution layer of your model
    intermediate_model = tf.keras.models.Model([model.inputs], [convOutput])
    
    # Perform forward pass to get feature maps for the input image
    feature_maps = intermediate_model.predict(imgArray)
    
    classWeights = finalDenseLayerWeights
    # Weighted sum of the feature maps
    cam = np.dot(feature_maps[0], classWeights)
    
    # Normalize the CAM
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    
    # Resize the CAM to the input image size (240x240)
    cam = cv2.resize(cam, (240, 240))
    return cam


# show the CAM for the images
# get just one batch of test images and break
for imgs, labs in testDataBatched.take(1):
    images = np.array(imgs)
    labels = np.array(labs)
    break


noImages = len(labels)
classes = ['Cat', 'Dog']
fig, axes = plt.subplots(noImages, 1, figsize=(5, noImages * 4))
fig.tight_layout()
plt.axis('off')


for i in range(noImages):
    img = images[i]
    label = labels[i]
    # Preprocess the image
    imgArray = preProcessImage(img)
    # Predict the class
    prediction = model.predict(imgArray)
    predictedClass = int(prediction[0] > 0.5)
    cam = generateClassicalCAM(model, imgArray, predictedClass)
    
    # Display the original image with CAM heatmap
    imgResized = cv2.resize(img, (240, 240))
    imgResized = np.uint8(imgResized * 255)
    # Convert the CAM to a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # Convert BGR to RGB
    # Overlay heatmap on the original image
    superimposed_img = cv2.addWeighted(imgResized, 0.6, heatmap, 0.4, 0)
    # Display the result
    print(classes[predictedClass])
    axes[i].imshow(superimposed_img[:, :, ::-1])  # Convert BGR to RGB for display
    axes[i].set_title(classes[predictedClass])
    axes[i].axis('off')

# Save the figure
plt.savefig('cam_results.png', bbox_inches='tight', dpi=150)
print("\nCAM visualization saved as 'cam_results.png'")
# plt.show()

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Step 1: Data Preparation
# Load the MNIST dataset
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
xTrain = xTrain.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0

# Convert labels to categorical format
yTrain = to_categorical(yTrain, num_classes=10)
yTest = to_categorical(yTest, num_classes=10)

# Step 2: Model Creation
# Initialize the model
model = Sequential()

# Add layers to the model
model.add(Flatten(input_shape=(28, 28)))  # Flatten the input
model.add(Dense(128, activation='relu'))   # Hidden layer with 128 neurons
model.add(Dense(10, activation='softmax')) # Output layer with 10 classes

# Step 3: Model Compilation
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 4: Model Training
model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_split=0.2)

# Step 5: Model Evaluation
test_loss, test_accuracy = model.evaluate(xTest, yTest)
print(f'Test accuracy: {test_accuracy:.4f}')

# Step 6: Prediction
predictions = model.predict(xTest)

# Display the first 5 test images and their predicted labels
for i in range(5):
    plt.imshow(xTest[i], cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(yTest[i])}')
    plt.axis('off')
    plt.show()

import tensorflow as t
import matplotlib.pyplot as mt 
import os 
import cv2 
import numpy as np
import subprocess
import time


mnist = t.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# One-Hot Encoding:
y_train_one_hot = t.keras.utils.to_categorical(y_train)
y_test_one_hot = t.keras.utils.to_categorical(y_test)

#----------------------------------------------------------------------------------------------------------#

                                         # Build the CNN model

model = t.keras.models.Sequential()

model.add(t.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu', input_shape=(28,28,1)))
model.add(t.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(t.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(t.keras.layers.Flatten())
model.add(t.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train,y_train_one_hot, validation_data=(x_test,y_test_one_hot), epochs=10)
model.save("check.keras")

#---------------------------------------------------------------------------------------------------------#

# Model loading

model = t.keras.models.load_model("check.keras")
loss, accuracy = model.evaluate(x_test, y_test_one_hot)
print(loss)
print(accuracy)


output_folder = "Digits_Folder"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def open_paint():

    subprocess.Popen(['mspaint'])  # Open MS Paint
    time.sleep(2)  # Wait a little for Paint to open

def capture_and_predict(image_no):

    img = cv2.imread(f"{output_folder}/{image_no}.png") [:,:,0]
    img = np.invert(np.array([img]))  
    prediction = model.predict(img)  # Model prediction
    print(f"This digit is probably{np.argmax(prediction)}")  #Maximum Probabilty
    
    # Show 
    mt.figure("Finally my project is done")
    mt.title("Number Identification")
    mt.imshow(img[0], cmap=mt.cm.binary)
    mt.show()

# Continuous loop to allow user to keep drawing
image_no = 0

while True:
    
    open_paint()  
    
    print("Please draw a digit in MS Paint and save it.")
    print(f"Make Completely sure that You have saved {image_no}.png")
    
    input("Press Enter after closing MS Paint...") #Asks User to enter input
   
    capture_and_predict(image_no) #Open CV and prediction of digit
    
    # Ask the user if they want to continue
    continue_choice = input("Do you want to continue? (y/n): ")
    
    if continue_choice.lower() != 'y':
        print("Exiting the program.")
        break  
    
    image_no += 1  

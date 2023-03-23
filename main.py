import os
import shutil
import random
from sklearn . model_selection import train_test_split
# Define the paths to the original data and the new train / test
split
original_dataset_dir = ’Original dataset ’
base_dir = ’Training dataset ’
train_dir = os. path . join ( base_dir , ’train ’)
test_dir = os. path . join ( base_dir , ’test ’)
# Create the s u b d i r e c t o r i e s for cats and dogs in the train / test
directories
train_cats_dir = os. path . join ( train_dir , ’cats ’)
train_dogs_dir = os. path . join ( train_dir , ’dogs ’)
test_cats_dir = os. path . join ( test_dir , ’cats ’)
test_dogs_dir = os. path . join ( test_dir , ’dogs ’
# Copy the cat and dog images to the new directories and split
into train / test sets
cat_filenames = os. listdir (os. path . join ( original_dataset_dir , ’
Cat ’))
dog_filenames = os. listdir (os. path . join ( original_dataset_dir , ’
Dog ’))
random . shuffle ( cat_filenames )
random . shuffle ( dog_filenames )
train_cat_filenames = cat_filenames [:1000]
train_dog_filenames = dog_filenames [:1000]
test_cat_filenames = cat_filenames [1000:1500]
test_dog_filenames = dog_filenames [1000:1500]
for filename in train_cat_filenames :
src = os. path . join ( original_dataset_dir , ’Cat ’, filename )
dst = os. path . join ( train_cats_dir , filename )
shutil . copyfile (src , dst )
for filename in train_dog_filenames :
src = os. path . join ( original_dataset_dir , ’Dog ’, filename )
dst = os. path . join ( train_dogs_dir , filename )
shutil . copyfile (src , dst )
for filename in test_cat_filenames :
src = os. path . join ( original_dataset_dir , ’Cat ’, filename )
dst = os. path . join ( test_cats_dir , filename )
shutil . copyfile (src , dst )
for filename in test_dog_filenames :
src = os. path . joi
# Define the image size and batch size
img_size = 224
batch_size = 32
# Create an I m a g e D a t a G e n e r a t o r object to perform data
augmentation
train_datagen = ImageDataGenerator (
rescale =1./255 ,
shear_range =0.2 ,
zoom_range =0.2 ,
horizontal_flip = True
)
# Create an I m a g e D a t a G e n e r a t o r object for the testing data
test_datagen = ImageDataGenerator ( rescale =1./255
# Load the training data
train_data = train_datagen . flow_from_directory (
train_dir ,
target_size =( img_size , img_size ) ,
batch_size = batch_size ,
class_mode =’categorical ’
)
# Load the testing data
test_data = test_datagen . flow_from_directory (
test_dir ,
target_size =( img_size , img_size ) ,
batch_size = batch_size ,
class_mode =’categorical ’
)
# Get the training images and labels
train_images , train_labels = train_data . next ()
# Get the testing images and labels
test_images , test_labels = test_data . next ()
# Split the data into training and testing sets
train_images , test_images , train_labels , test_labels =
train_test_split (
train_images , train_labels , test_size =0.2 , random_state =42)
# Convert the labels to one - hot encoding
train_labels = tf. keras . utils . to_categorical ( train_labels ,
num_classes =2)
test_labels = tf. keras . utils . to_categorical ( test_labels ,
num_classes =2)
                                   from tensorflow . keras . applications import MobileNetV2
from tensorflow . keras . layers import Dense , Flatten
from tensorflow . keras . models import Model
from tensorflow . keras . optimizers import Adam
import PIL
import numpy as np
# Load the pre - trained MobileNetV2 model
base_model = MobileNetV2 ( input_shape =( img_size , img_size , 3) ,
include_top =False , weights =’imagenet ’)
# Freeze the base model layers
for layer in base_model . layers :
layer . trainable = False
# Add a new c l a s s i f i c a t i o n layer
x = base_model . output
x = Flatten () (x)
x = Dense (128 , activation =’relu ’)(x)
predictions = Dense (2 , activation =’softmax ’)(x)
# Define the new model
model = Model ( inputs = base_model .input , outputs = predictions )
train_labels = np. where ( train_labels > 0.5 , 1, 0)
test_labels = np. where ( test_labels > 0.5 , 1, 0)
# Compile the model
model . compile ( optimizer = Adam ( learning_rate =0.0001) , loss =’
binary_crossentropy ’, metrics =[ ’accuracy ’])
# Train the model
history = model . fit (
train_data ,
steps_per_epoch = train_data . samples // batch_size ,
epochs =10 ,
validation_data = test_data ,
validation_steps = test_data . samples // batch_size
)
                                   # Evaluate the model on the testing set
loss , accuracy = model . evaluate ( test_data , verbose =1)
print (’Test accuracy :’, accuracy )
print (’Test loss :’, loss )
import numpy as np
from tensorflow . keras . preprocessing . image import load_img ,
img_to_array
# Load a sample image
img_path = ’cat2 .jpg ’
img = load_img ( img_path , target_size =( img_size , img_size ))
img_array = img_to_array ( img )
img_array = np. expand_dims ( img_array , axis =0)
# Use the model to make a prediction
prediction = model . predict ( img_array )
if prediction [0][0] > prediction [0][1]:
print (’The image is a cat !’)
else :
print (’The image is a dog !’)

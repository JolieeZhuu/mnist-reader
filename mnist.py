# imports
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# DATA --------------------------------------------------------------------------------------------------------------------------------
# based on documentation: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
training_data = datasets.MNIST(
    root = "data", # test_data will be in /data folder
    train = True, # extract the training data (rather than testing data)
    transform = transforms.ToTensor(), # originally PIL image, but pixels from 0-255 are converted to 0.0-1.0
    download = True
) # btw this is a function lol
# training_data is of type Dataset

testing_data = datasets.MNIST(
    root = "data",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# print(training_data, end="\n\n")
# print("training_data size", len(training_data))
# print("sample", training_data[0][0])
# print("label", training_data[0][1])

# separate training_data input (x) from training_data output (y)
x_train = training_data.data
y_train = training_data.targets
# print(x_train) # input data
# print(y_train) # output data

x_test = testing_data.data
y_test = testing_data.targets

# visualization using matplotlib
num_classes = 10 # technically i wanted to print out 0 to 9 inclusive but it just took the first 10 samples
# fig, ax = plt.subplots(nrows = 1, ncols = num_classes, figsize = (15, 15)) # use the parameter variables themselves to specify their values

# for i in range(num_classes):
#     sample_data = x_train[y_train == i][0] # second index refers to the label; we are seeing what each image of the digit looks like (hence y_train == i)
#     ax[i].imshow(sample_data.squeeze(), cmap='gray') # gray scale doesn't require a third argument, so squeeze removes dimensions of size 1
#     ax[i].set_title("Output: {}".format(training_data[i][1]), fontsize=16)

# plt.subplots_adjust(wspace = 0.75) # adjust horizontal spacing
# plt.show()

# manually categorize the outputs (both training and testing) into vectors
print('before vectorizing')
for i in range(num_classes):
    print(y_train[i]) # before vectorizing

# make an equivalent keras.utils.to_categorically() function with pytorch
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
for i in range(num_classes):
    print(y_train[i]) # after vectorizing

# normalize/prepare data
# print(x_train[0]) # check before normalizing
"""instead of data ranging from 0-255, it will range from 0-1
"""
x_train = x_train / 255
y_train = y_train / 255

# print(x_train[0]) # check after normalizing

# flattening/reshaping the data from 3D (6000, (28, 28)) to 2D (6000, 784)
#print(x_train[0].shape) # check the structure of the matrix

x_train = x_train.reshape(x_train.shape[0], -1) # from np; retrieves size of dataset (6000)
# -1 tells np to calculate the size of each image (28*28=784)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape)
#--------------------------------------------------------------------------------------------------------------------------------------

# Model -------------------------------------------------------------------------------------------------------------------------------

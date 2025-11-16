# imports
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# data
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
"""
WHERE DO YOU FIND THIS THOUGH???
"""
x_train = training_data.data
y_train = training_data.targets
print(x_train) # input data
print(y_train) # output data

# visualization using matplotlib

# example
# fig, ax = plt.subplots()             # Create a figure containing a single Axes.
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.
# plt.show()                           # Show the figure.

num_cols = 10 # technically i wanted to print out 0 to 9 inclusive but it just took the first 10 samples
fig, ax = plt.subplots(nrows = 1, ncols = num_cols, figsize = (15, 15)) # use the parameter variables themselves to specify their values
# try changing figsize to see what happens

for i in range(num_cols):
    sample_data = training_data[i][0] # second index refers to the label
    ax[i].imshow(sample_data.squeeze(), cmap='gray') # gray scale doesn't require a third argument, so squeeze removes dimensions of size 1
    ax[i].set_title("Output: {}".format(training_data[i][1]), fontsize=16)

plt.subplots_adjust(wspace = 0.75) # adjust horizontal spacing
plt.show()

# manually categorize the outputs into vectors
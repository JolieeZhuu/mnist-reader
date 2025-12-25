"""MNIST Handwritten Digit Recognizer
Only PyTorch
1. Import all PyTorch related libraries
2. Fetch dataset and split it into training 
and testing (inputs and outputs)
3. (Optional) Visualize sample of dataset
4. Vectorize, normalize, and prepare data
5. Create a very simple neural network
6. Define how you'll train the model
(identifying loss function and optimizer)
7. Train the model
"""

# Also note to self that interpreter on VSCode
# should be the .venv (torch and torchvision
# are installed here)

# imports
import torch
import torch.nn as nn # for the neural network
import torch.optim as optim # for the optimizer
from torchvision import datasets, transforms # MNIST in datasets, transforms for preprocessing
from torch.utils.data import DataLoader # purpose: batching and shuffling

# hyperparameters (for training)
batch_size = 64 # images per gradient step
learning_rate = 0.001 # step size in parameter space
epochs = 5 # cycles of passing through full dataset

training_data = datasets.MNIST(
    root="data", # place data into /data folder
    train=True, # extract training data
    transform=transforms.ToTensor(), # originally PIL images, but we scaled it from [0,255] to [0.0, 1.0]
    download=True
)

testing_data = datasets.MNIST(
    root="data", # place data into /data folder
    train=False, # extract testing data
    transform=transforms.ToTensor(), # originally PIL images, but we scaled it from [0, 255] to [0.0,1.0]
    download=True
)

# shuffle dataset to prevent learning order bias
# also puts data into batches of <batch_size>
# why do we use batches? 
# - faster training
# - smoother gradients (adds gradient noise)
# batch_size = 1 -> stochastic gradient descent (SGD)
training_loader = DataLoader(
    dataset=training_data,
    batch_size=batch_size,
    shuffle=True
)

testing_loader = DataLoader(
    dataset=testing_data,
    batch_size=batch_size,
    shuffle=False
)

# define architecture of nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 128) # fetches flattened input, returns matrix of length 128
        self.layer2 = nn.Linear(128, 64) # get that matrix of 128, returns matrix of 64
        self.layer3 = nn.Linear(64, 10) # get that matrix of 64, returns 10 logits (0-9)
    # why 128 and 64? apparently it's pretty fast, but you could try 256 -> 128 -> 64 -> 32 -> 10
    # or 100 -> 50 -> 10...anything to get you from 784 -> ... -> 10
    
    # forward pass
    def forward(self, x):
        # normalize the data
        x = x.view(x.size(0), -1) # equivalent to NumPy's [0, 255] -> [0, 1], flatten (28,28) -> 784
        
        # relu because it's faster and requires less processing
        x = torch.relu(self.layer1(x)) # add non-linearity to processed data; relu(x) = max(0, x)
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x # returns logits, which are vectors of unnormalized predictions

# define model
model = NeuralNetwork()

# define loss function
criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss is standard loss for multi-class classification
# since we have 10 classes (0-9)
# Input: logits for a softmax function (of each batch)
# softmax converts logits to actual probabilities
# then we take the negative log, and return the average loss of each batch

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# will update the weights using gradients
# model.parameters() tells us which weights to update

# train model for <epochs> times
for epoch in range(epochs):
    total_loss = 0
    for images, labels in training_loader: # since data is structured in [[images], [labels]] form
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels) # compute the loss
        optimizer.zero_grad() # clears accumulated gradients from previous iteration
        loss.backward() # performs backward propagation to find optimal weights
        optimizer.step() # updates weights using gradients
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}") # prints loss per data sample
        total_loss += loss.item() * images.size(0) # multiply by batch size
    avg_loss = total_loss / len(training_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}") # prints average loss

# evaluating model (checking for accuracy) - no learning
model.eval() # switch to evaluation mode (originally in training mode)

correct = 0
total = 0

# forward pass once
with torch.no_grad(): # disables gradient tracking
    for images, labels in testing_loader: # since data is structured in [[images], [labels]] form
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # picks index of largest logit (the predicted class from 0-9)
        total += labels.size(0) # counts total samples processed
        correct += (predicted == labels).sum().item() # counts correct predictions

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
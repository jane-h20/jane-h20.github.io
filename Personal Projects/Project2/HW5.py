# Jane Hawkins
# CNN Using UC Merced Land Use Dataset

#exec(open("HW5.py").read())

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

classes = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 
           'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 
           'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 
           'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 
           'storagetanks', 'tenniscourt') # can I make this easier??
# agricultural is class 0, airplane is class 1, etc.

def create_paths_label():
    # e.g. path = './data/UCMerced_LandUse/Images/'
    #label = 'agricultural'
    #pth = './data/UCMerced_LandUse/Images/agricultural/agriculture00.tif'

    training_image_paths = []
    training_labels = []
    testing_image_paths = []
    testing_labels = []

    for i in range(len(classes)): # loop through all 21 classes
        
        # loop through first 50 images for training data
        for j in range(49): 
            my_string_j = str(j)
            if len(my_string_j) == 1:
                my_string_j = '0' + my_string_j

            pth = './data/UCMerced_LandUse/Images/' + classes[i] + '/' + classes[i] + my_string_j + '.tif'
            training_image_paths.append(pth)
            training_labels.append(i)
    
        # Repeat for testing data
        for j in range(49, 100): # loop throuhg first 50 images for training data
            my_string_j = str(j)
            if len(my_string_j) == 1:
                my_string_j = '0' + my_string_j

            pth = './data/UCMerced_LandUse/Images/' + classes[i] + '/' + classes[i] + my_string_j + '.tif'
            testing_image_paths.append(pth)
            testing_labels.append(i)
    
    return (training_image_paths, training_labels, testing_image_paths, testing_labels)

from custom_dataset import CustomImageDataset # importing my class from a different file
my_transform = transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Get Paths and Labels
(training_image_paths, training_labels, testing_image_paths, testing_labels) = create_paths_label()

# Get an instance of my new class:
train_dataset = CustomImageDataset(training_image_paths, training_labels, transform=my_transform)
test_dataset = CustomImageDataset(testing_image_paths, testing_labels, transform=my_transform)

my_batch_size = 4
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=my_batch_size, shuffle=True, num_workers=1)

### END DATA LOADING ###

import matplotlib.pyplot as plt
import numpy as np
# define a function to be able to visualize some of these images
def imshow(img):
    img = img / 2 + 0.5 # un-normalize the image
    npimg = img.numpy() # turn into numpy object
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
data_iter = iter(trainloader)
images, labels = next(data_iter) # retrieve next item from iterator

# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(my_batch_size)))
# show images
imshow(torchvision.utils.make_grid(images))


# Set up CNN
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 4)
        self.fc1 = nn.Linear(12 * 61 * 61, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 21)

    def forward(self, x): # send an image through CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)

        # Now send through FFNN
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x
    

my_cnn = Net() # get an instance of this class

# Start Training:
import torch.optim as optim

criterion = nn.CrossEntropyLoss() # this will define the loss
optimizer = optim.SGD(my_cnn.parameters(), lr=0.001, momentum=0.9)

# Set total number of epochs
N = 10

for epoch in range(N):  # loop over entire dataset N times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = my_cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
PATH = './UC_Merced_net.pth'
torch.save(my_cnn.state_dict(), PATH)

# Test the network on the test data
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
# load back in our saved model
my_cnn.load_state_dict(torch.load(PATH, weights_only=True))

# have it guess the output
outputs = my_cnn(images) #The outputs are energies for the 10 classes.

# let’s get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# look at how the network performs on the whole dataset
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = my_cnn(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')

# what are the classes that performed well, and the classes that did not perform well:
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = my_cnn(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

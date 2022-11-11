# Saaketh Narayan
# handwritten digit recognition example

import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(os.path.dirname(os.path.abspath(
    __file__)), download=True, train=True, transform=transform)
testset = datasets.MNIST(os.path.dirname(os.path.abspath(
    __file__)), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

# first layer size 784 # 28x28 images
# second layer size 128
# third layer size 64
# output layer size 10 (softmax probabilities over all classes -- digits)
layers = [784, 128, 64, 10]

model = nn.Sequential(
    nn.Linear(layers[0], layers[1]),
    nn.ReLU(),
    nn.Linear(layers[1], layers[2]),
    nn.ReLU(),
    nn.Linear(layers[2], layers[3]),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 16

for e in range(epochs):
    # track running loss over all epochs
    running_loss = 0
    for images, labels in trainloader:
        # flatten images
        images = images.view(images.shape[0], -1)

        # zero out gradients, we are not accumulating gradients over batches
        optimizer.zero_grad()

        # output of this NN is the log probability of each class
        output = model(images)
        # compute negative log likelihood loss on output and true labels
        loss = criterion(output, labels)

        # backpropagate the loss to model parameters
        loss.backward()

        # update weights based on backpropagated loss (gradients)
        optimizer.step()

        # increment running loss
        running_loss += loss.item()

    # printed loss here is per batch. length of trainloader is # total samples / batchsize = # batches
    print("Epoch {} - Training Loss: {}".format(e+1, running_loss/len(trainloader)))
    print("Elapsed Training Time (min) =", (time()-time0)/60, "\n")

""" images, labels = next(iter(testloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logps = model(img)

# e^(log prob) = prob
ps = torch.exp(logps)
probs = list(ps.numpy()[0])
print("Predicted Digit = ", probs.index(max(probs)))
print(probs)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(images[0].numpy().squeeze(), cmap='gray_r')
axs[1].bar(np.arange(10), probs)
axs[1].set_xticks(np.arange(10), np.arange(10, dtype='int'))
plt.show() """

correct = 0
total = 0

for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)

        # deactivate autograd engine
        with torch.no_grad():
            logps = model(img)

        # e^(log prob) = prob
        ps = torch.exp(logps)
        probs = list(ps.numpy()[0])
        pred_label = probs.index(max(probs))
        true_label = labels.numpy()[i]

        # update correct count if true equals predicted
        if (true_label == pred_label):
            correct += 1
        total += 1

print("Total number of testing images:", total)
print("Model accuracy =", correct/total)

import json
import numpy as np
import os

import vectorizer

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class TikTokDataset(Dataset):
    
    def __init__(self, data):
        self.videoPosts = []
        self.interaction = []

        for video_properties, interaction in data:
            self.videoPosts.append(video_properties)
            self.interaction.append(interaction)

        self.videoPosts = torch.stack(self.videoPosts)
        self.interaction = torch.stack(self.interaction)

        self.numInstances = len(self.videoPosts)

    def __len__(self):
        return self.numInstances

    def __getitem__(self, index):
        return self.videoPosts[index], self.interaction[index]

class Net(nn.Module):
    def __init__(self, size_in, size_hidden, size_out):
        super().__init__()
        self.linLayer1 = nn.Linear(size_in, size_hidden)
        self.linLayer2 = nn.Linear(size_hidden, size_hidden)
        self.linLayer3 = nn.Linear(size_hidden, size_hidden)
        self.linLayer4 = nn.Linear(size_hidden, size_out)
        self.batchNormLayer1 = nn.BatchNorm1d(size_hidden)
        self.batchNormLayer2 = nn.BatchNorm1d(size_out)
        self.dropLayer = nn.Dropout(0.3)
        self.activationFunction = nn.ReLU()
        self.lossFunction = nn.L1Loss()

    def forward(self, v):
        x = self.linLayer1(v)
        x = self.activationFunction(x)
        x = self.dropLayer(x)
        x = self.linLayer2(x)
        x = self.activationFunction(x)
        x = self.dropLayer(x)
        x = self.linLayer3(x)
        x = self.activationFunction(x)
        x = self.linLayer4(x)
        out = self.activationFunction(x)
        return out

    def loss(self, out, gt):
        return self.lossFunction(out, gt)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

def train_model(epoch_count, net, train_data_loader, validation_data_loader):
    optimizer = optim.SGD(net.parameters(), lr = 0.002, momentum = 0.005)
    for epoch in range(epoch_count):
        print("Epoch " + str(epoch) + " out of " + str(epoch_count))
        one_training_epoch(net, train_data_loader, optimizer)
        validation(net, validation_data_loader, optimizer)
    print("Training Complete")
    return

def one_training_epoch(net, data_loader, optimizer):
    net.train()
    running_loss = 0.0
    correct_instances = 0
    total_instances = 0
    error_multiplier_threshold = 0.6
    
    for i, data in enumerate(data_loader, 0):
        inputs, expected_outputs = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = net.loss(outputs, expected_outputs)
        loss.backward()
        optimizer.step()

        magnitude_expected = torch.norm(expected_outputs.type(torch.FloatTensor)).item()
        acceptable_loss_lower_bound = magnitude_expected*error_multiplier_threshold
        acceptable_loss_upper_bound = magnitude_expected*(2 - error_multiplier_threshold)
        
        if loss.item() >= acceptable_loss_lower_bound and loss.item() <= acceptable_loss_upper_bound:
            correct_instances += 1
        total_instances += 1
            
        running_loss += loss.item()

        if (i % 100 == 0):
            print("Prediction Accuracy: " + str(correct_instances/total_instances))
            print("Running Loss: " + str(running_loss))

    print("Training completed")
    print("Final Prediction Accuracy: " + str(correct_instances/total_instances))
    print("Final Running Loss: " + str(running_loss))

def validation(net, data_loader, optimizer):
    net.eval()
    running_loss = 0.0
    correct_instances = 0
    total_instances = 0
    error_multiplier_threshold = 0.6

    for i, data in enumerate(data_loader, 0):
        inputs, expected_outputs = data
        outputs = net(inputs)
        
        magnitude_expected = torch.norm(expected_outputs.type(torch.FloatTensor)).item()
        acceptable_loss_lower_bound = magnitude_expected*error_multiplier_threshold
        acceptable_loss_upper_bound = magnitude_expected*(2 - error_multiplier_threshold)

        loss = net.loss(outputs, expected_outputs)
        
        if loss.item() >= acceptable_loss_lower_bound and loss.item() <= acceptable_loss_upper_bound:
            correct_instances += 1
        total_instances += 1
            
        running_loss += loss.item()

    running_loss /= total_instances
    percent_accuracy = correct_instances/total_instances

    print("Average Loss: " + str(running_loss))
    print("Percent Accuracy: " + str(percent_accuracy))

def data_loaders(training_data, validation_data, batch_size=1):

    dataset = TikTokDataset(training_data + validation_data)

    training_indices = [i for i in range(len(training_data))]
    validation_indices = [i for i in range(len(training_data), len(training_data) + len(validation_data))]

    training_shuffler = SubsetRandomSampler(training_indices)
    training_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=training_shuffler)
    
    validation_shuffler = SubsetRandomSampler(validation_indices)
    validation_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_shuffler)

    return training_data_loader, validation_data_loader
         

file = open("trending.json")
data = json.load(file)
file.close()

training_data, validation_data = vectorizer.getTrainingAndValidationDataAsTorchTuples(data)

training_loader, validation_loader = data_loaders(training_data, validation_data)

size_in = 9 # Combining all features from caption, author, audio
size_hidden = 256
size_out = 4

num_epochs = 6

Model = Net(size_in, size_hidden, size_out).to("cpu")
train_model(num_epochs, Model, training_loader, validation_loader)
Model.save("neuralNet.pth")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_data = datasets.FashionMNIST(root='Data', train = True, download=True,transform=transforms.ToTensor()) #mnist is the dataset
#root is where the fiels are, train true selects the training images, tensor tunrs them to tensor of numbers from 0 to 1

test_data = datasets.FashionMNIST(root='Data', train = False,download = True,transform=transforms.ToTensor()) #false selects the test images
train_loader = DataLoader(train_data,batch_size=64,shuffle=True) #shuffle mixes the order for all epochs so it doesnt learn from the ordering and batch goes through 64 images, more memory friendly

test_loader = DataLoader(test_data,batch_size=1000) #1000 batch cuz order doesnt matter, 1000 to go faster, more images per step

#baseline model
model = nn.Sequential(nn.Flatten(), #parameters, flattens the image into one row of numbers 28x28
nn.Linear(784,128 ), #linear layer expects a row and not a grid, learnable weights
nn.ReLU(),# changes negative numbers to 0, 
nn.Linear(128,10) #last layer outputs 10 classes with respective score. guess is the biggest
)




#loss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#training loop
for epoch in range(3):
    total = 0
    for x,y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out,y)
        loss.backward()
        optimizer.step()
        total+= loss.item()
    print (epoch +1,total/len(train_loader))


#testing loop
correct = 0
for x, y in test_loader:
    out = model(x)
    pred = out.argmax( dim = 1)
    correct += (pred == y).sum().item()#replaces the previous prints
     #look at one batch w break, to go through 10 its the accuracy loop

accuracy = correct/len(test_data)
print(accuracy)

pred = out.argmax(dim=1)
print(pred.shape)
print(pred[:5])
print(y[:5])


#counting snippet
print((pred == y)[:5])
print((pred == y).sum().item()) #makes new tensor and check if outputs are true with preidictions

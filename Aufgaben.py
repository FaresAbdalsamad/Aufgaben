#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


w1 = 0.5
w2 = 0.3
b = 0.1

tu_train = np.linspace(-10, 10, 1000)  
y_train = w2 * tu_train**2 + w1 * tu_train + b  

tu_train = torch.tensor(tu_train, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)


# In[3]:


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.linear1 = nn.Linear(1, 10) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1) 

    def forward(self, tu):
        x = self.linear1(tu)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# In[4]:


def train(model, tu_train, y_train, num_epochs=5000, learning_rate=0.01):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(num_epochs):
        
        outputs = model(tu_train)
        loss = criterion(outputs, y_train)

       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


# In[5]:


model = NewModel()
losses = train(model, tu_train, y_train)


epochs = range(1, len(losses) + 1)


# In[6]:


plt.plot(epochs, losses)
plt.xlabel('Epochen')
plt.ylabel('Loss')
plt.title('Loss-Verlauf des neuen Modells')


# In[ ]:





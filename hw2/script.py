
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class policeNN(nn.Module):
    def __init__(self,input_dim, output_dim, scope, n_layers,
                 hide_dim,dropout_prob,activation, output_activation=None,):
        super(self,policeNN).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.n_layers=n_layers
        self.hide_dim=hide_dim
        self.dropout_prob=dropout_prob
        if activation not in ['relu','tanh']:
            raise NameError('activation function is not exist')
        if output_activation not in ['relu', 'tanh','softmax']:
            raise NameError('output_activation function is not exist')
        self.activation=activation
        self.output_activation=output_activation
        self.layers=[]
        self.input_layer=nn.Sequential(nn.Linear(self.input_dim,self.hide_dim),nn.ReLU())
        for i in range(1,n_layers):
            self.layers.append(nn.Sequential(nn.Linear(self.input_dim,self.hide_dim),nn.ReLU()))
        self.outlayer=nn.Sequential(nn.Linear(self.hide_dim,self.output_dim),nn.ReLU())
        self.dropout=nn.Dropout(p=self.dropout_prob)
    def forward(self,x):
        x=torch.tensor(x)
        #y=torch.tensor(y)
        out=self.input_layer(x)
        for i in range(len(self.layers)):
            out=self.layers[i](out)
        out=self.outlayer(out)
        out=self.dropout(out)
        return out
data=np.array([[1,2],[2,3],[3,3]])
sns.set(style="darkgrid")

# Load an example dataset with long-form data
#fmri = sns.load_dataset("fmri")



x = np.linspace(0, 15, 1)
print(x.shape)
print(np.sin(x).shape,np.random.rand(10, 1).shape,np.random.randn(10, 1).shape)
data = np.sin(x) + np.random.rand(1, 31)
print(data.shape)
print(pd.DataFrame(data).head())


sns.tsplot(data = data,
           color = 'g'
           )

#plt.show()
plt.savefig('baseline.jpg')
# Plot the responses for different events and regions

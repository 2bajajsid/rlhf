import numpy as np
from scipy.stats import ortho_group
from scipy.special import expit
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import math

rng = np.random.default_rng()
np.random.seed(946)

# Step 1: Simulate data 
# 1. Covariates x using a fixed MVN distribution
# 2. Responses y_1 and y_2
n_0 = 10
Q = ortho_group.rvs(n_0)

# eigenvalues must be positive because  
# variance co-variance is positive 
# semi-definite
eigen_vals = rng.uniform(low = 0.0, high = 10.0, size = n_0)
D = np.diag(eigen_vals)

Sigma = Q @ D @ Q.T
Mu = rng.uniform(low = -5.0, high = 5.0, size = n_0)

# generate responses: 
# 1. y_1 using linear regression 
# 2. y_2 using logistic regression
N = 10000
X = torch.tensor(np.random.multivariate_normal(mean = Mu, cov = Sigma, size = N), 
                 dtype=torch.float32)
beta_star = torch.tensor(np.random.normal(loc = 0.0, scale = 1.0, size = n_0),
                         dtype=torch.float32)
y_1 = torch.zeros(N)
y_2 = torch.zeros(N)

for i in range(N):
    y_1[i] = np.random.normal(beta_star @ X[i, :], 1)
    y_2[i] = np.random.binomial(n = 1, p = expit(beta_star @ X[i, :]))

# Step 2: 
# Implement mu_p 

# One hidden layer
class mu_p_network(torch.nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super().__init__()
        self.linear_1_layer = nn.Linear(input_size, hidden_neurons)
        self.linear_2_layer = nn.Linear(hidden_neurons, hidden_neurons)
        self.linear_3_layer = nn.Linear(hidden_neurons, output_size)

        nn.init.normal_(self.linear_1_layer.weight, 
                       mean = 0.0, 
                       std = 1 / np.sqrt(input_size))
        
        nn.init.normal_(self.linear_2_layer.weight, 
                       mean = 0.0, 
                       std = 1 / np.sqrt(hidden_neurons))
        
        nn.init.normal_(self.linear_3_layer.weight, 
                       mean = 0.0, 
                       std = 1 / np.sqrt(hidden_neurons))
        
        self.n_0 = input_size
        self.n = hidden_neurons
        print("Network initialized!")
    
    # Prediction Function
    def forward(self, x):
        self.layer_in = self.linear_1_layer(x)
        self.act = torch.sigmoid(self.layer_in)
        self.layer_2 = self.linear_2_layer(self.act)
        self.act = torch.sigmoid(self.layer_2)
        self.layer_out = self.linear_3_layer(self.act)
        return self.layer_out
    
size = 20 # size of hidden layers
model = mu_p_network(n_0, size, n_0)
batches_per_epoch=50
        
def loss_linear(y, beta, x):
    loss = torch.tensor(0.0, requires_grad = True)
    for i in range(batches_per_epoch):
        loss = loss + (y[i] - (torch.matmul(beta[i], x[i])))**2
    return loss

def loss_logistic(y, beta, x):
    loss = torch.tensor(0.0, requires_grad = True)
    for i in range(batches_per_epoch):
        loss = loss + (y[i] - expit(torch.matmul(beta[i], x[i])))**2
    return loss

# training loop
optimizer = torch.optim.SGD(model.parameters(),
                            lr = 0.001 * size)
epochs=500
cost=[]
total=0

for epoch in range(epochs):
    total = 0
    epoch = epoch + 1

    for i in range(math.ceil(N / batches_per_epoch)):
        random_batch = np.random.choice(N, batches_per_epoch)
        X_batch = X[random_batch]
        Y_batch = y_1[random_batch]

        beta_hat = model(X_batch.float())
        optimizer.zero_grad()

        loss = loss_linear(Y_batch, beta_hat, X_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() 
        cost.append(total / batches_per_epoch)

    if epoch % 1000 == 0:
        print(str(epoch)+ " " + "epochs done!")  
        
plt.plot(cost)
plt.xlabel('Epochs')
plt.title('Loss')
plt.show()
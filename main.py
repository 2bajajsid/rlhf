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

sigma_norm = 1
for i in range(N):
    y_1[i] = np.random.normal(beta_star @ X[i, :], sigma_norm**2)
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
    
size = 10000 # size of hidden layers
model = mu_p_network(n_0, size, n_0)
batches_per_epoch=250
        
def loss_linear(y, beta, x):
    loss = torch.tensor(0.0)
    for i in range(batches_per_epoch):
        loss = loss + (y[i] - (torch.matmul(beta[i], x[i])))**2
    return loss

# Step 3: Train the sft network
def loss_logistic(y, beta, x):
    loss = torch.tensor(0.0)
    for i in range(batches_per_epoch):
        loss = loss + (y[i] - expit(torch.matmul(beta[i], x[i])))**2
    return loss

# training loop
optimizer = torch.optim.SGD(model.parameters(),
                            lr = 0.00001 * size)
max_norm = 1.0

epochs=5000
cost=[]
total=0

for epoch in range(epochs):
    total = 0

    random_batch = np.random.choice(N, batches_per_epoch)
    X_batch = X[random_batch]
    Y_batch = y_1[random_batch]

    beta_hat = model(X_batch.float())
    optimizer.zero_grad()

    loss = loss_linear(Y_batch, beta_hat, X_batch)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    total += loss.item() 
    cost.append(total / batches_per_epoch)

    if epoch % 250 == 0:
        print(str(epoch)+ " " + "epochs done!")
        print(cost[epoch - 1])

    epoch = epoch + 1
        
plt.plot(cost)
plt.xlabel('Epochs')
plt.ylabel('Loss / data-point')
plt.title('Loss / data-point vs. Epochs')
plt.show()

# Step 4: Create a reward function
# sample some prompts x
# define ground-truth function for response y
# train reward function r_phi to maximize difference in reward
def pdf(x, y, beta_star = beta_star): 
    pdf_eval = torch.tensor(0.0)
    y_true = torch.matmul(beta_star, x)
    normal_const = 1 / np.sqrt(2 * math.pi * (sigma_norm)**2)
    pdf_eval = pdf_eval + torch.tensor(normal_const * (math.exp(- (1 / 2) * ((y - y_true) / sigma_norm)**2)))
    return pdf_eval

X = torch.tensor(np.random.multivariate_normal(mean = Mu, cov = Sigma, size = N), 
                 dtype=torch.float32)
beta_hat = model(X.float())

# each row is (y_w, y_l)
Y = torch.tensor(np.zeros(shape = (N,2)))
for i in range(0, N):
    y_1_prime = np.random.normal(beta_hat[i, :].detach().numpy() @ X[i, :].detach().numpy(), 1)
    y_2_prime = np.random.normal(beta_hat[i, :].detach().numpy() @ X[i, :].detach().numpy(), 1)
    if (pdf(X[i, :], y_1_prime) > pdf(X[i, :], y_2_prime)):
        Y[i, 0] = y_1_prime
        Y[i, 1] = y_2_prime
    else:
        Y[i, 0] = y_2_prime
        Y[i, 1] = y_1_prime

reward_model = mu_p_network(n_0 + 1, size, 1)
batches_per_epoch=500
        
# Step 5: Train the reward model
def loss_preferred_not_preferred(r_y_w, r_y_l):
    loss = torch.tensor(0.0)
    for i in range(batches_per_epoch):
        loss = loss - torch.log(torch.sigmoid(torch.sub(r_y_w[i], r_y_l[i])))
    return loss

# training-loop
optimizer = torch.optim.SGD(reward_model.parameters(),
                            lr = 0.0001 * size)
max_norm = 1.0

epochs=5000
cost=[]
total=0

for epoch in range(epochs):
    total = 0

    random_batch = np.random.choice(N, batches_per_epoch)
    X_batch = X[random_batch]
    Y_batch = Y[random_batch]
    
    input_model = np.zeros(shape = (2 * batches_per_epoch, 1 + n_0))
    
    input_model[0:batches_per_epoch, 0] = Y_batch[:,0]
    input_model[0:batches_per_epoch, 1:] = X_batch
    
    input_model[batches_per_epoch:, 0] = Y_batch[:,1]
    input_model[batches_per_epoch:, 1:] = X_batch

    input = torch.from_numpy(input_model).float()
    rewards = reward_model(input)

    optimizer.zero_grad()

    loss = loss_preferred_not_preferred(rewards[0:batches_per_epoch], 
                                        rewards[batches_per_epoch:])
    loss.backward()

    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm)
    optimizer.step()
    total += loss.item() 
    cost.append(total / batches_per_epoch)

    if epoch % 250 == 0:
        print(str(epoch)+ " " + "epochs done!")
        print(cost[epoch - 1])

    epoch = epoch + 1
        
plt.plot(cost)
plt.xlabel('Epochs')
plt.ylabel('Loss / data-point')
plt.title('Loss / data-point vs. Epochs (Reward Model)')
plt.show()

# Step 6: Alignment-Model 
# Proximal Policy Operation (PPO)
def loss_ppo(beta_theta, beta_sft, x):
    kl_penalty = 1
    kl_divergence = torch.tensor(0.0)
    loss = torch.tensor(0.0)
    rewards_acc = torch.tensor(0.0)

    input_model = torch.zeros([batches_per_epoch, 1 + n_0])
    for i in range(batches_per_epoch):
        input_model[i, 1:] = x[i,:]
        input_model[i, 0] = beta_theta[i,:] @ x[i,:]
    rewards = reward_model(input_model)
    
    for i in range(batches_per_epoch):
        kl_divergence = kl_divergence + ((1/2) * kl_penalty * (torch.pow(torch.norm(torch.sub(beta_theta[i,:], beta_sft[i,:])), 2)))
        rewards_acc = rewards_acc + rewards[i]

    loss = rewards_acc - kl_divergence
    return -1 * loss

aligned_model = mu_p_network(n_0, size, n_0)
aligned_model.load_state_dict(model.state_dict())
optimizer = torch.optim.SGD(aligned_model.parameters(),
                            lr = 0.0001 * size)

epochs=5000
cost=[]
total=0

for epoch in range(epochs):
    total = 0

    random_batch = np.random.choice(N, batches_per_epoch)
    X_batch = X[random_batch]
    Y_batch = y_1[random_batch]

    beta_theta = aligned_model(X_batch.float())
    beta_sft = model(X_batch.float())
    optimizer.zero_grad()

    loss = loss_ppo(beta_theta, beta_sft, X_batch)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(aligned_model.parameters(), max_norm)
    optimizer.step()
    total += loss.item() 
    cost.append(total / batches_per_epoch)

    if epoch % 250 == 0:
        print(str(epoch)+ " " + "epochs done!")
        print(cost[epoch - 1])

    epoch = epoch + 1
        
plt.plot(cost)
plt.xlabel('Epochs')
plt.ylabel('Loss / data-point')
plt.title('Loss / data-point vs. Epochs (Aligned Model)')
plt.show()
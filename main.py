
import numpy as np
from drn import DRN
import config
import time
from datetime import datetime

import torch
import torch.nn as nn

import sys
import os

current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")

n_in, n_layers, n_nodes, n_out = config.n_in, config.n_layers, config.n_nodes, config.n_out
q, hidden_q = config.q, config.hidden_q

model_weights_filename = config.model_weights_filename

# training data
data_dir = config.data_dir
train_x = np.loadtxt(data_dir+'train_x.dat')
train_y = np.loadtxt(data_dir+'train_y.dat')
test_x = np.loadtxt(data_dir+'test_x.dat')
test_y = np.loadtxt(data_dir+'test_y.dat')
train_x = torch.as_tensor(train_x[:config.Ntrain].reshape((-1, 1, q)),dtype=torch.float32)
train_y = torch.as_tensor(train_y[:config.Ntrain].reshape((-1, 1, q)),dtype=torch.float32)
test_x = torch.as_tensor(test_x[:config.Ntest].reshape((-1, 1, q)),dtype=torch.float32)
test_y = torch.as_tensor(test_y[:config.Ntest].reshape((-1, 1, q)),dtype=torch.float32)


# DRN fully-connected network

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
	DRN(n_in,q,n_nodes,hidden_q),
	DRN(n_nodes,hidden_q,n_out,q)
)

model.to(device)

# define loss criterion
criterion = torch.nn.MSELoss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

if os.path.isfile(model_weights_filename):
	state_dict = torch.load(model_weights_filename)
	model.load_state_dict(state_dict['model_state_dict'])
	optimizer.load_state_dict(state_dict['optimizer_state_dict'])
	print('weights loaded successfully!!!\n{}'.format(model_weights_filename))

num_batches = int(train_x.shape[0]/config.batch_size)

t0 = time.time()
for epoch in range(config.Nepoch):
	avg_mse_loss = 0.0

	# Loop over all batches
	model.train()
	for i in range(0, train_x.shape[0], config.batch_size):
		j = i + config.batch_size
		if j <= train_x.shape[0]:
			x=train_x[i:j]
			y=train_y[i:j]

			x = x.to(device)
			y = y.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			y_logits = model(x)
			loss = criterion(y_logits, y)
			loss.backward()
			optimizer.step()

			avg_mse_loss += loss.item()

	if epoch % 1 == 0:
		model.eval()
		with torch.no_grad():
			x=test_x
			y=test_y

			x = x.to(device)
			y = y.to(device)

			y_logits = model(x)
			loss = criterion(y_logits, y)

			print('epoch: ' + str(epoch) + ', train mse: ' + str(avg_mse_loss / num_batches)  + ', test mse: ' + str(loss.item()))
print ('training time: ' + str(time.time() - t0))

# evaluate train and test
model.eval()
with torch.no_grad():
	x=train_x
	y=train_y

	x = x.to(device)
	y = y.to(device)

	y_logits = model(x)
	loss = criterion(y_logits, y)
	ltrain_mse_ = loss.item()

	print('ltrain mse: ' + str(ltrain_mse_))

	x=test_x
	y=test_y

	x = x.to(device)
	y = y.to(device)

	y_logits = model(x)
	loss = criterion(y_logits, y)
	ltest_mse_ = loss.item()

	print('ltest mse: ' + str(ltest_mse_))


model_weights_filename = "trained_model_weights" + current_time + '__' + str(epoch+1) + ".pth"
state_dict = {	'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)






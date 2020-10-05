run_id = 5
seed = 0
Nepoch = 100
learning_rate = 0.1
batch_size = 10
load_W_from_file = False

#NW architecture: Nhidden layers of Nnodes each, ip and op 1 node
# n_in - [n_layers] x n_nodes - n_out
n_in = 1
n_layers = 1
n_nodes = 5
n_out = 1

data_dir = './OU_q100/'
q = 100
hidden_q = 10
Ntrain = 200
Ntest = 100

# previously saved model
model_weights_filename = ""
# model_weights_filename = "trained_model_weights__2020_10_05__15_08_43__100.pth"

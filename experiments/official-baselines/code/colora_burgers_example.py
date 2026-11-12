# Code cells of the official CoLoRA burgers.ipynb (commit 7367004), concatenated; IPython magics and gif-plot cells dropped.
from jax import vmap
import jax.numpy as jnp
from pathlib import Path
import numpy as np
from einops import rearrange
import jax
import matplotlib.pyplot as plt

from colora.data import load_all_hdf5, split_data_by_mu, prepare_coordinate_data
data_dir = Path('./data')
data_path = data_dir / 'burgers'
mus, sols, spacing = load_all_hdf5(data_path)

train_mus = np.asarray([0.001, 0.00199, 0.00298, 0.00496, 0.00595, 0.00694, 0.00892, 0.01])
test_mus = np.asarray([0.00397, 0.00793])
train_sols, test_sols = split_data_by_mu(mus, sols, train_mus, test_mus) # mus X variables X time X space_x X space_y
n_mu_train, n_q, n_t, n_x1, n_x2 = train_sols.shape
n_mu_test, n_q, n_t, n_x1, n_x2 = test_sols.shape
time = spacing[1]
x_space = spacing[2]
y_space = spacing[3]

print(f'n_mu train: {n_mu_train}, n_mu test: {n_mu_test}, n_variables: {n_q}, n_time samples: {n_t}, n_x samples: {n_x1}, n_x2 samples: {n_x2}')


y_train, mu_t_train, X_grid =  prepare_coordinate_data(spacing, train_mus, train_sols)
y_test, mu_t_test, X_grid =  prepare_coordinate_data(spacing, test_mus, test_sols)

def normalize(x, mean, std):
    return (x-mean)/std

mean, std = jnp.mean(mu_t_train, axis=0), jnp.std(mu_t_train, axis=0)
mu_t_train = normalize(mu_t_train, mean, std)
mu_t_test = normalize(mu_t_test, mean, std)


from colora.build import build_colora

key = jax.random.PRNGKey(1)

x_dim = 2
mu_t_dim = 2
u_dim = 1

d_len_x = x_space[-1] - x_space[0]
d_len_y = y_space[-1] - y_space[0]


u_layers = ['P', 'C', 'C', 'C', 'C', 'C', 'C', 'C'] # seven colora layers with 1 alpha each means we will have laten dim of 7
h_layers = ['D', 'D', 'D']
rank = 3

u_hat_config = {'width': 25, 'layers': u_layers}
h_config = {'width': 15, 'layers': h_layers}

u_hat_fn, h_fn, theta_init, psi_init = build_colora(
    u_hat_config, h_config, x_dim, mu_t_dim, u_dim, lora_filter=['alpha'],period=[d_len_x, d_len_y], rank=rank, key=key)


h_v_mu_t = vmap(h_fn, in_axes=(None, 0)) # vmap over mu_t array to generate array of phis
u_hat_v_x =  vmap(u_hat_fn, in_axes=(None, None, 0)) # vmaped over x to generate solution field over space points
u_hat_v_x_phi =  vmap(u_hat_v_x, in_axes=(None, 0, None)) # vmaped over x to generate solution field over space points

def predict(psi_theta, mu_t, X_grid):
    psi, theta = psi_theta
    phis = h_v_mu_t(psi, mu_t)
    pred = u_hat_v_x_phi(theta, phis, X_grid)
    return pred

def relative_loss_fn(psi_theta, mu_t, sols, X_grid):
    pred = predict(psi_theta, mu_t, X_grid)
    loss = jnp.linalg.norm(
        sols - pred, axis=(1,2)) / jnp.linalg.norm(sols, axis=(1,2))
    return loss.mean()



from colora.data import Dataset

# the dataset is just responsible for batching the data over the mu_t tensor 
# in order to aviod memory overflow on the GPU
dataset = Dataset(mu_t_train, X_grid, y_train, n_batches=15, key=key)
dataset = iter(dataset)
def args_fn():
    return next(dataset)

from colora.adam import adam_opt

psi_theta = (psi_init, theta_init)
opt_psi_theta, loss_history = adam_opt(psi_theta, relative_loss_fn, args_fn, steps=100_000, learning_rate=5e-3, verbose=True)
opt_psi, opt_theta = opt_psi_theta



pred = predict(opt_psi_theta, mu_t_test, X_grid)
pred = rearrange(pred, '(M T) (N1 N2) Q -> M Q T N1 N2', Q=n_q, T=n_t, N1=n_x1, N2=n_x2)


test_vec =  rearrange(test_sols, 'M Q T N1 N2 -> M (Q T N1 N2)')
pred_vec =  rearrange(pred, 'M Q T N1 N2 -> M (Q T N1 N2)')
rel_err = np.linalg.norm(test_vec- pred_vec, axis=1)/np.linalg.norm(test_vec, axis=1)
mean_rel_err = rel_err.mean()
print(f'Test mean relative error: {mean_rel_err:.2E}')

phis = h_v_mu_t(opt_psi, mu_t_test)
phis = rearrange(phis, '(M T) D -> M T D', T=n_t)




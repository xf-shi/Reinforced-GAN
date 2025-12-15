# Deep Learning Algorithms for Equilibrium Models 

Training Pipeline for 

- Reinforced-GAN
- Combo Solver
- Deep FBSDE Solver

This repository provides a flexible training pipeline to train generator, discriminator, and combined models for multi-agent dynamic hedging problems. The code is built in PyTorch and supports multiple training configurations, including deep hedging, FBSDE, and leading-order approximations.

---

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```
2. Install the dependencies:
```bash
pip3 install torch numpy matplotlib tqdm pytz
```

## Usage
The specification of equilibrium solver to train as well as hyperparameters are all achieved at the `training_args` variable. Here are a few important flags that can be applied to the settings all equilibrium solvers:
- `use_true_mu`: If set to `True`, then the drift is computed by using an analytical expression in terms of the volatility and hedging strategies. Note that this is only supported for quadratic loss or when the number of agents is 2. If set to `False`, then the drift is parameterized by the neural network.
- `clearing_known`: If set to `True`, then the neural network only parameterizes the hedging strategies $`\dot{\varphi}_1, \dots, \dot{\varphi}_{N-1}`$ of the first `N-1` agents, whereas the last agent's strategy is assumed to clear the market:

$$
\dot{\varphi}_N = -\sum_{n=1}^{N-1} \dot{\varphi}_n
$$

If set to `False`, then the neural network will parameterize the hedging strategies of all $N$ agents.

### Training Reinforced-GAN

Set `use_combo: False`. The following hyperparameters will be used for training:
- `total_rounds`: The total number of rounds of iterative updates of the generator and the discriminator, where in each round, we first train the generator for a specified number of epochs, then we train the discriminator for a specified number of epochs.
- `gen_hidden_lst`: The number of neurons in the hidden layers of the generator.
- `gen_lr`: The list of initial learning rates for the generator for each round. If the length of the list is smaller than the `total_rounds`, then the last element in the learning rate list will be used for all remaining rounds.
- `gen_decay`: The decay rate of generator learning rate.
- `gen_scheduler_step`: The frequency of the generator learning rate decay.
- `gen_solver`: The optimizer for training the generator.
- `gen_epoch`: The number of training epochs of generator for each round.
- `gen_sample`: The number of training samples used for the generator.
- `train_gen`: If set to `True`, then the generator will be retrained. If set to `False`, then we will load the latest trained generator and freeze it.
- `dis_hidden_lst`: The number of neurons in the hidden layers of the discriminator.
- `dis_lr`: The list of initial learning rates for the discriminator for each round. If the length of the list is smaller than the `total_rounds`, then the last element in the learning rate list will be used for all remaining rounds.
- `dis_decay`: The decay rate of discriminator learning rate.
- `dis_scheduler_step`: The frequency of the discriminator learning rate decay.
- `dis_solver`: The optimizer for training the discriminator.
- `dis_epoch`: The number of training epochs of discriminator for each round.
- `dis_sample`: The number of training samples used for the discriminator.
- `train_dis`: If set to `True`, then the generator will be retrained. If set to `False`, then we will load the latest trained discriminator and freeze it.

### Training Combo Solver
Set `use_combo: True` and `use_fbsde: False`. The following hyperparameters will be used for training:
- `combo_hidden_lst`: The number of neurons in the hidden layers of the combo solver.
- `combo_lr`: The list of initial learning rates for the combo solver.
- `combo_decay`: The decay rate of combo solver learning rate.
- `combo_scheduler_step`: The frequency of the combo solver learning rate decay.
- `combo_solver`: The optimizer for training the combo solver.
- `combo_epoch`: The number of training epochs of combo solver.
- `combo_sample`: The number of training samples used for the combo solver.

### Training FBSDE Solver
Set `use_combo: True` and `use_fbsde: True`. The following hyperparameters will be used for training:
- `combo_hidden_lst`: The number of neurons in the hidden layers of the fbsde solver.
- `combo_lr`: The list of initial learning rates for the fbsde solver.
- `combo_decay`: The decay rate of fbsde solver learning rate.
- `combo_scheduler_step`: The frequency of the fbsde solver learning rate decay.
- `combo_solver`: The optimizer for training the fbsde solver.
- `combo_epoch`: The number of training epochs of fbsde solver.
- `combo_sample`: The number of training samples used for the fbsde solver.

## Comparison Results
### Quadratic Trading Cost, 10 Agents
| Method | $\sum_{n\in\mathcal{N}} J_n(\dot{\varphi}_n)$ | $\left\|\sum_{n\in\mathcal{N}}\dot{\varphi}_n\right\|^2$ | $\left\|S_T^\theta - \mathcal{S}\right\|^2$ | $S_0$ | $\mu_0$ | $\sigma_0$ |
|--------|---------------------------------------------|----------------------------------------------------------|-----------------------------------------------|-------|--------|------------|
| Ground Truth | $-2.08\times10^{-1}$ | $2.11\times10^{-13}$ | $2.95\times10^{-17}$ | $3.61\times10^{-1}$ | $2.16\times10^{-1}$ | $1.25\times10^{0}$ |
| Reinforced-GAN (Mu Known) | $-2.09\times10^{-1}$ | $2.21\times10^{-3}$ | $2.32\times10^{-5}$ | $3.58\times10^{-1}$ | $2.23\times10^{-1}$ | $1.26\times10^{0}$ |
| Reinforced-GAN (Mu Unknown) | $-2.09\times10^{-1}$ | $2.30\times10^{-5}$ | $2.73\times10^{-7}$ | $3.61\times10^{-1}$ | $1.36\times10^{-1}$ | $1.26\times10^{0}$ |
| Combo (Mu Known) | $-4.86\times10^{-2}$ | $7.43\times10^{-1}$ | $2.61\times10^{0}$ | $-1.10\times10^{0}$ | $6.17\times10^{-1}$ | $2.11\times10^{0}$ |
| Combo (Mu Unknown) | $-2.07\times10^{-1}$ | $1.91\times10^{-2}$ | $6.05\times10^{-6}$ | $3.44\times10^{-1}$ | $5.54\times10^{0}$ | $1.29\times10^{0}$ |
| FBSDE (Mu Known) | $-3.83\times10^{-1}$ | $2.41\times10^{-4}$ | $2.19\times10^{-1}$ | $0.00\times10^{0}$ | $2.26\times10^{-1}$ | $1.28\times10^{0}$ |
| FBSDE (Mu Unknown) | $-3.84\times10^{-1}$ | $2.56\times10^{-3}$ | $1.39\times10^{-1}$ | $0.00\times10^{0}$ | $1.61\times10^{0}$ | $1.27\times10^{0}$ |
**Table:** Comparison of Reinforced-GANs against ground truth for 10 agents with quadratic costs.   Simulation is performed using 3000 sample paths.

### Superlinear Trading Cost, 2 Agents
| Method | $\sum_{n\in\mathcal{N}} J_n(\dot{\varphi}_n)$ | $\left\|\sum_{n\in\mathcal{N}}\dot{\varphi}_n\right\|^2$ | $\left\|S_T^\theta - \mathcal{S}\right\|^2$ | $S_0$ | $\mu_0$ | $\sigma_0$ |
|--------|---------------------------------------------|----------------------------------------------------------|-----------------------------------------------|-------|--------|------------|
| Leading Order | $-8.94\times10^{-4}$ | $1.89\times10^{-11}$ | $7.47\times10^{-3}$ | $4.15\times10^{-1}$ | $7.66\times10^{-1}$ | $1.07\times10^{0}$ |
| Reinforced-GAN (Mu Known) | $-3.74\times10^{-4}$ | $1.07\times10^{-2}$ | $2.71\times10^{-5}$ | $4.46\times10^{-1}$ | $7.86\times10^{-1}$ | $1.09\times10^{0}$ |
| Reinforced-GAN (Mu Unknown) | $-5.62\times10^{-4}$ | $2.19\times10^{-6}$ | $1.30\times10^{-5}$ | $4.58\times10^{-1}$ | $7.75\times10^{-1}$ | $1.08\times10^{0}$ |
| Combo (Mu Known) | $2.61\times10^{0}$ | $1.17\times10^{2}$ | $6.59\times10^{0}$ | $-4.53\times10^{1}$ | $1.24\times10^{-5}$ | $4.31\times10^{-3}$ |
| Combo (Mu Unknown) | $-2.24\times10^{-3}$ | $6.78\times10^{-2}$ | $8.08\times10^{-8}$ | $5.44\times10^{-1}$ | $9.05\times10^{-1}$ | $1.19\times10^{0}$ |
| FBSDE (Mu Known) | NaN | NaN | NaN | NaN | NaN | NaN |
| FBSDE (Mu Unknown) | NaN | NaN | NaN | NaN | NaN | NaN |
**Table:** Comparison of Reinforced-GANs against ground truth for 2 agents with $3/2$-power costs.   Simulation is performed using 3000 sample paths.

### Superlinear Trading Cost, 10 Agents
| Method | $\sum_{n\in\mathcal{N}} J_n(\dot{\varphi}_n)$ | $\left\|\sum_{n\in\mathcal{N}}\dot{\varphi}_n\right\|^2$ | $\left\|S_T^\theta - \mathcal{S}\right\|^2$ | $S_0$ | $\mu_0$ | $\sigma_0$ |
|--------|---------------------------------------------|----------------------------------------------------------|-----------------------------------------------|-------|--------|------------|
| Frictionless | $-4.80\times10^{-1}$ | $2.30\times10^{-9}$ | $2.91\times10^{-2}$ | $2.02\times10^{-1}$ | $1.39\times10^{-1}$ | $1.00\times10^{0}$ |
| GAN (Mu Unknown) | $-9.46\times10^{-2}$ | $9.32\times10^{-5}$ | $5.04\times10^{-6}$ | $3.65\times10^{-1}$ | $1.64\times10^{-1}$ | $1.10\times10^{0}$ |
| Combo (Mu Unknown) | $-3.76\times10^{-1}$ | $8.63\times10^{1}$ | $3.25\times10^{-7}$ | $3.61\times10^{-1}$ | $2.42\times10^{-1}$ | $1.32\times10^{0}$ |
| FBSDE (Mu Unknown) | NaN | NaN | NaN | NaN | NaN | NaN |
**Table:** Comparison of Reinforced-GANs against ground truth for 10 agents with $3/2$-power costs.  Simulation is performed using 3000 sample paths.

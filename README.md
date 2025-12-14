# Neural Network Training Pipeline for Reinforced-GAN, Combo Solver, and FBSDE Solver

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
- `clearing_known`: If set to `True`, then the neural network only parameterized the hedging strategies $\dot{\varphi}_1, \dots, \dot{\varphi}_{N-1}$ of first `N-1` agents, whereas the last agent's strategy is assumed to clear the market, i.e., $\dot{\varphi}_N = -\sum_{n = 1}^{N-1} \dot{\varphi}_n$. If set to `False`, then the neural network will parameterize the hedging strategies of all N agents.

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

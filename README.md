# Model-based RL for Building HVAC Control

This repository is the official implementation of [Building HVAC Scheduling Using Reinforcement Learning via Neural Network Based Model Approximation](https://arxiv.org/abs/1910.05313). 

## Requirements

To install requirements:
### Python package
```setup
pip install -r requirements.txt
```

### EnergyPlus
Please follow https://github.com/IBM/rl-testbed-for-energyplus and install EnergyPlus version 9.1.0

## Training

To run the PID agent, run

```train
python train_pid.py --city SF
```

To train the PPO agent, run

```train
python train_ppo.py --city SF
```

To train the model-based RL with random shooting (RS), run

```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 100
```

To train the model-based RL with dagger, run

```train
python train_model_based.py --city SF --mpc_horizon 5 --num_days_on_policy 10 --training_epochs 100 --dagger
```

It will create a folder called ``runs`` that includes all the state, action and rewards during the training.
The EnergyPlus generated files will be in the ``log`` folder.

### Available cities
- SF
- Golden
- Chicago
- Sterling

We also provide shell script file in case you want to run everything. Checkout
- run_pid.sh
- run_ppo.sh
- run_model_based_plan.sh
- run_model_based_dagger.sh

## Citation
```bib
@article{Zhang2019BuildingHS,
  title={Building HVAC Scheduling Using Reinforcement Learning via Neural Network Based Model Approximation},
  author={Chi Zhang and S. Kuppannagari and R. Kannan and V. Prasanna},
  journal={Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  year={2019}
}
```
Please also cite the paper that introduces the environment
```bib
@InProceedings{10.1007/978-981-13-2853-4_4,
author="Moriyama, Takao and De Magistris, Giovanni and Tatsubori, Michiaki and Pham, Tu-Hoa and Munawar, Asim and Tachibana, Ryuki",
title="Reinforcement Learning Testbed for Power-Consumption Optimization",
booktitle="Methods and Applications for Modeling and Simulation of Complex Systems",
year="2018",
publisher="Springer Singapore",
address="Singapore",
pages="45--59",
isbn="978-981-13-2853-4"
}
```
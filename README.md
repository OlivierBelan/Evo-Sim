# Evo-Sim
**Evo-Sim** is a comprehensive neuroevolutionary simulator designed to support both Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs). It integrates a variety of sophisticated algorithms tailored for evolutionary simulation and neural network experimentation.

## Supported Algorithms
- GA (Genetic Algorithm)
- NEAT (NeuroEvolution of Augmenting Topologies)
- HyperNetwork (HyperNEAT if used with NEAT but can be used with any other algorithms)
- MAP-ELITE (Multi-dimensional Archive of Phenotypic Elites)
- NSLC (Novelty Search with Local Competition)
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- NES (Natural Evolution Strategies)
- OpenES (OpenAI Evolution Strategies)
- ARS (Augmented Random Search)
- NSGA (Non-dominated Sorting Genetic Algorithm) - (coming soon)
- ES-HyperNEAT - (coming soon)

**Note:** Comprehensive documentation will be available soon.

## Installation

### Create a virtual environment
It is recommended to use a virtual environment for installation to manage dependencies effectively. Follow these steps to set up your environment and install the required packages:

```bash
python3 -m venv env/
source env/bin/activate
```
### Installing Dependencies
Execute the script below to install all necessary packages on Linux or macOS:

```bash
bash install_linux_mac.sh
```


## Usage
To start using Evo-Sim, you can run a test example with the following command:
```bash
python test_SL.py --algo NEAT --problem WINE --nn SNN --nb_runs 2 --nb_generation 200 --record False --cpu 20
```
or 
```bash
python test_RL.py --problem HalfCheetah --algo NES-evosax --nn SNN --nb_runs 2 --nb_generation 200 --nb_episodes 1 --record False --cpu 20
```


## Troubleshooting
### Common Issue: PyBullet Environment Error


Common Issue: PyBullet Environment Error
If you encounter an ```AttributeError``` related to the ```pybullet``` environment during installation or execution:
```AttributeError: 'dict' object has no attribute 'env_specs'```

Here is how to resolve it:

Locate and modify the `pybullet_envs/__init__.py` file as follows:

#### Original Code:
```python
def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)
```
#### Modified Code:
```python
def register(id, *args, **kvargs):
  if id in registry:
    return
  else:
    return gym.register(id, *args, **kvargs)
```
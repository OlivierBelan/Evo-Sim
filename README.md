# Evo-Sim
A NeuroEvolutionary Simulator - Compatible with Spiking Neural Networks  &amp; Artificial Neural Networks

Algorithm: GA, NEAT, HyperNEAT, MAP-ELITE, Novelty-Search Local Competion (NSLC), CMA-ES, NES, OpenES, ARS, NSGA and ES-HyperNEAT (coming soon).

A complete documentation is coming very soon.


To test the program, execute either python test_SL.py NES-evosax or python test_RL.py NES-evosax from within the test folder. Additional options are available in the test_SL.py and test_RL.py files, as well as within the configuration files located in the config folder.

To facilitate comparisons, an ANN runner built with PyTorch is also available. To use it, uncomment the line start_config_path = "./config/config_ann/SL/" in either test_SL.py or test_RL.py, and comment out the corresponding SNN configuration line start_config_path = "./config/config_snn/SL/"

During the installation/run of pybullet env (QD_env), if you encounter the following error:
```AttributeError: 'dict' object has no attribute 'env_specs'```
You can fix it by changing the file ```pybullet_envs/__init__.py``` the section:
```python
def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)
```
to:
```python
def register(id, *args, **kvargs):
  if id in registry:
    return
  else:
    return gym.register(id, *args, **kvargs)
```
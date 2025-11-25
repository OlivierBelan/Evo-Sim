# Evo-Sim

**Evo-Sim** is a **neuroevolutionary simulator** designed to support both **Spiking Neural Networks (SNNs)** and **Artificial Neural Networks (ANNs)**.
It integrates a wide range of evolutionary algorithms and provides a flexible framework for RL and SL experiments on CPU and GPU.

A lighter, SNN-only runner is also available in **SNN-sim**: [https://github.com/OlivierBelan/SNN-Sim](https://github.com/OlivierBelan/SNN-Sim), while Evo-Sim is the more complete framework (more algorithms, encoders/decoders, and tools for NeuroEvolution).

---

## Features

* Support for **SNNs and ANNs**
* Clock-based SNN simulator with **LIF neurons**
* **CPU and GPU** backends (Cython / CUDA where supported)
* Rich set of **NeuroEvolution algorithms** (GA, NEAT, MAP-Elites, NES, CMA-ES, etc.)
* Multiple **encoders/decoders** for spiking and non-spiking networks
* Flexible network architectures (feedforward, skip connections, richer topologies)
* Separate **Reinforcement Learning (RL)** and **Supervised Learning (SL)** pipelines

---

## Supported Algorithms

Evo-Sim currently supports or plans to support the following algorithms:

* **GA** (Genetic Algorithm)
* **NEAT** (NeuroEvolution of Augmenting Topologies)
* **HyperNetwork** (HyperNEAT-style when used with NEAT, but can be used with other algorithms)
* **MAP-ELITE** (Multi-dimensional Archive of Phenotypic Elites)
* **NSLC** (Novelty Search with Local Competition)
* **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
* **NES** (Natural Evolution Strategies)
* **OpenES** (OpenAI Evolution Strategies)
* **ARS** (Augmented Random Search)
* **NSGA** (Non-dominated Sorting Genetic Algorithm) – *coming soon*
* **ES-HyperNEAT** – *coming soon*

---

## Installation

This repository uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies.

First, install `uv` (for example):

```bash
pip install uv
```

(or any other method you prefer to install `uv`).

Then, from the repository root, install the dependencies with:

```bash
bash setup.sh
```
This will install the required Python packages and set up the environment for Evo-Sim.

If you want to use the **GPU backends**, you also need a working CUDA toolchain and compatible GPU drivers on your system (at least `nvcc` available for local compilation).

---

## Configuration files

For **SNN + NES** runs (similar to SNN-sim), all simulation, SNN and optimisation hyperparameters are stored in two main config files:

* `NES_CONFIG_RL.cfg` – configuration for **Reinforcement Learning** experiments
* `NES_CONFIG_SL.cfg` – configuration for **Supervised Learning** experiments

These config files centralise, among other things:

* **Simulation settings**

  * Total run time and time step (`dt`)
  * Run-time margins
  * Runner type (RL vs supervised), online vs offline modes

* **SNN architecture and topology**

  * Number of inputs, hidden layers and outputs
  * Layer sizes (e.g. `H1:32`, `H2:32`, …)
  * Connectivity patterns (`I->H1`, `H1->O`, `I->O`, recurrent links, etc.)

* **Encoding / decoding choices**

  * Encoder type (e.g. poisson, binomial, exact, rate, combinatorial, latency, direct, derivative, etc.)
  * Decoder type (e.g. rate, voltage, augmented, …)

* **Optimisable parameters for NeuroEvolution**

  * Which parameters are evolved (e.g. `params_to_update = weight` or also thresholds, voltages, delays, etc.)
  * Ranges and distributions for each parameter type (min/max, μ, σ, decays, …)

* **NES and optimisation settings**

  * Population size, initial sigma, temperature, mean decay
  * Optimisation objective (maximize, minimize, closest_to_zero)

Other algorithms (NEAT, GA, CMA-ES, MAP-Elites, etc.) may use additional or separate configuration options, but the goal is the same: **modify the config files to change experiments, not the core code**.

---

## Usage

All examples can be launched via `uv` (recommended) or directly with `python`. Below are a few example commands from the `test` directory.

### Supervised Learning example (NEAT)

From the `test` directory:

```bash
uv run test_SL.py --algo NEAT --problem WINE --nn SNN --nb_runs 2 --nb_generation 200 --record False --device cpu --nb_cpu 20
```

This will run a supervised learning experiment (e.g. WINE classification) using:

* The NEAT algorithm (`--algo NEAT`)
* A spiking (or selected) neural network (`--nn SNN` or other)
* The settings defined in the corresponding config file(s)

---

### Reinforcement Learning example (NES with SNN)

From the `test` directory:

**CPU only:**

```bash
uv run test_RL.py --problem HalfCheetah --algo NES-evosax --nn SNN --nb_runs 2 --nb_generation 200 --nb_episodes 1 --record False --device cpu --nb_cpu 20 --config config/config_snn/NES_CONFIG_RL.cfg
```

**With GPU:**

```bash
uv run test_RL.py --problem HalfCheetah --algo NES-evosax --nn SNN --nb_runs 2 --nb_generation 200 --nb_episodes 1 --record False --device gpu --nb_gpu 1 --config config/config_snn/NES_CONFIG_RL.cfg
```

These commands will:

* Load the RL configuration from `NES_CONFIG_RL.cfg`
* Build the SNN defined in the config (architecture, neuron model, parameters)
* Encode observations according to the chosen encoder and observation type
* Decode SNN activity into actions via the selected decoder and action scaling
* Optimise the chosen SNN parameters with NES across generations

You can swap `--nn SNN` for an ANN model (if configured) and switch `--algo` to other supported algorithms (GA, CMA-ES, MAP-Elites, etc.) depending on your experiment.

---

## Customisation

* To modify **simulation length**, time step or runner behaviour: edit the corresponding fields in the relevant config file(s) (e.g. `NES_CONFIG_RL.cfg`, `NES_CONFIG_SL.cfg`, or algorithm-specific configs).
* To change the **network architecture** (number of layers, sizes, connectivity): update the appropriate architecture/“genome” sections in the configs.
* To decide which SNN parameters are **optimised by NES** (weights only vs full SNN parameters): adjust `params_to_update` and the parameter sections in `NES_CONFIG_*` files.
* To experiment with different **algorithms**: switch the `--algo` flag (e.g. `GA`, `NEAT`, `CMA-ES`, `OpenES`, `MAP-ELITE`, `NSLC`, etc.) and use the corresponding config presets.
* To test different **encoding/decoding strategies** for SNNs: change the `encoder` and `decoder` entries and their associated parameters in the config files.

The design goal is that you can run new experiments and ablations by **only editing config files and command-line flags**, without modifying the core Evo-Sim implementation.

---

## Relationship to SNN-sim

* **SNN-sim** ([https://github.com/OlivierBelan/SNN-Sim](https://github.com/OlivierBelan/SNN-Sim)) is a focused, standalone SNN runner based on LIF neurons and NES.
* **Evo-Sim** is the **full framework**: it reuses and extends the SNN runner, and adds many more algorithms, ANN support, richer encoders/decoders, and additional tools for large-scale NeuroEvolution experiments.

If you only need a minimal SNN + NES setup, SNN-sim may be sufficient. If you want a general-purpose NeuroEvolution playground, Evo-Sim is the intended entry point.

---

## Citation

If you use Evo-Sim in your research, please cite:

> Olivier Belan, **"Evo-Sim: A neuroevolution framework for spiking and artificial neural networks"**, 2025.
> GitHub repository: [https://github.com/OlivierBelan/Evo-Sim](https://github.com/OlivierBelan/Evo-Sim)

```bibtex
@misc{belan2025evosim,
  author       = {Belan, Olivier},
  title        = {Evo-Sim: A neuroevolution framework for spiking and artificial neural networks},
  year         = {2025},
  howpublished = {\url{https://github.com/OlivierBelan/Evo-Sim}},
  note         = {GitHub repository}
}
```

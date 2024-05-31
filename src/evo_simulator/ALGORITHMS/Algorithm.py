
from evo_simulator.GENERAL.Attribute import Attribute_Paramaters
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Index_Manager import get_new_population_id
from typing import Dict, Any

class Algorithm:
    def __init__(self, config_path_file:str, name:str, extra_info:Dict[Any, Any] = None) -> None:
        self.name:str = name
        self.pop_size:int = None
        self.config_path_file:str = config_path_file
        self.extra_info:Dict[Any, Any] = extra_info

        # General
        self.population_manager:Population = Population(get_new_population_id(), config_path_file)
        self.attributes_manager:Attribute_Paramaters = Attribute_Paramaters(config_path_file)
        

    def run(self, global_population:Population) -> Population:
        raise NotImplementedError

    # def mutation_attributes_mu_sigma(self, population:Dict[int, Genome], mu_neuron_bias:float, sigma_neuron_coef:float, mu_synapse_bias:float, sigma_synapse_coef:float) -> None:
    #     self.mutation.attributes.neurons_mu_sigma(population, self.mu_neuron, self.sigma_neuron, self.min_neuron, self.max_neuron, mu_neuron_bias, sigma_neuron_coef)
    #     self.mutation.attributes.synapses_mu_sigma(population, self.mu_synapse, self.sigma_synapse, self.min_synapse, self.max_synapse, mu_synapse_bias, sigma_synapse_coef)

    # def mutation_attributes_sigma(self, population:Dict[int, Genome], mu_neuron_bias:float, sigma_neuron_coef:float, mu_synapse_bias:float, sigma_synapse_coef:float) -> None:
    #     self.mutation.attributes.neurons_sigma(population, self.sigma_neuron, self.min_neuron, self.max_neuron, mu_neuron_bias, sigma_neuron_coef)
    #     self.mutation.attributes.synapses_sigma(population, self.sigma_synapse, self.min_synapse, self.max_synapse, mu_synapse_bias, sigma_synapse_coef)

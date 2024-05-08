from evo_simulator.GENERAL.Genome import Genome
from evo_simulator.ALGORITHMS.Algorithm import Algorithm
from evo_simulator.GENERAL.Population import Population, Population_NN
from evo_simulator.GENERAL.Index_Manager import get_new_population_id
from evo_simulator.GENERAL.Archive import Archive

from problem.Problem import Problem

from typing import List, Dict, Tuple, Any
import evo_simulator.TOOLS as TOOLS
from evo_simulator.Record.Record import Record
import numpy as np
import time
import ray
import os
import shutil

class Neuro_Evolution:

    def __init__(self, nb_generations:int, nb_runs:int, is_record:bool, config_path:str, cpu:int, is_Archive:bool=False):
        
        self.config_path:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")

        # 1- Initialize configs
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(self.config_path, ["NEURO_EVOLUTION", "Genome_NN", "Runner_Info"])
        
        self.nb_generations = nb_generations if nb_generations != None else int(self.config["NEURO_EVOLUTION"]["nb_generations"])
        self.nb_runs = nb_runs if nb_runs != None else int(self.config["NEURO_EVOLUTION"]["nb_runs"])
        self.verbose = True if self.config["NEURO_EVOLUTION"]["verbose"] == "True" else False
        self.is_record:bool = is_record
        self.record:Record = None
        self.network_type:str = self.config["Genome_NN"]["network_type"]
        self.nb_hidden_neurons:int = TOOLS.hiddens_nb_from_config(self.config["Genome_NN"]["hiddens"])
        self.optimization_type:str = self.config["NEURO_EVOLUTION"]["optimization_type"] # maximize, minimize, closest_to_zero
        self.is_Archive:bool = is_Archive
        self.archives:List[Archive] = None
        self.cpu_affinity:bool = False if self.config["Runner_Info"].get("cpu_affinity", None) == None else True if self.config["Runner_Info"]["cpu_affinity"] == "True" else False
        if is_Archive == True:
            self.__get_archives_names(self.config_path)

        if self.network_type == "SNN":
            import snn_simulator.build as build
            build.build_simulator()
            self.check_snn_refractory_and_delay()
            self.run_length:int = int(self.config["Runner_Info"]["run_time"])
            self.run_with_margin:int = int(self.config["Runner_Info"]["run_time_margin"]) + self.run_length
            self.dt:float = float(self.config["Runner_Info"]["dt"])
            self.decay_method:str = self.config["Runner_Info"]["decay_method"]
            self.encoder:str = self.config["Runner_Info"]["encoder"]
            self.decoder:str = self.config["Runner_Info"]["decoder"]
            self.record_run:str = self.config["Runner_Info"]["record"]
            self.encoder_record_text:str = self.encoder
            self.decoder_record_text:str = self.decoder

            # Voltage Decoding
            self.is_voltage_encoding:bool = True if self.decoder == "voltage" else False
            if self.decoder == "voltage":
                self.config.update(TOOLS.config_function(self.config_path, ["Voltage_Decoder"]))
                self.is_output_threshold:bool = False if self.config["Voltage_Decoder"]["disable_output_threshold"] == "True" else True
                self.voltage_min:str = self.config["Voltage_Decoder"]["voltage_min"]
                self.voltage_max:str = self.config["Voltage_Decoder"]["voltage_max"]
                self.decoder_record_text:str = self.decoder + "_voltage_min-" + self.voltage_min + "_voltage_max-" + self.voltage_max
                self.max_possible_spikes:int = 0

            # Rate Decoding / Augmented Decoding
            if self.decoder == "rate":
                self.config.update(TOOLS.config_function(self.config_path, ["Rate_Decoder"]))
                self.decoding_ratio:float = float(self.config["Rate_Decoder"]["ratio_max_output_spike"])
                self.output_multiplier:int = int(self.config["Genome_NN"]["outputs_multiplicator"])
                self.max_possible_spikes:int = self.run_length * self.output_multiplier * self.decoding_ratio
                self.decoder_record_text:str = self.decoder + "_max_spikes-" + str(self.max_possible_spikes)

            if self.decoder == "augmented":
                self.config.update(TOOLS.config_function(self.config_path, ["Augmented_Decoder"]))
                self.max_possible_spikes:int = self.config["Augmented_Decoder"]["spike_max"]
                self.decoder_record_text:str = self.decoder + "_spike_max-" + str(self.max_possible_spikes)

            # Combinatorial Encoding
            self.is_combinatorial_encoding:bool = True if self.encoder == "combinatorial" else False
            if self.is_combinatorial_encoding == True:
                self.config.update(TOOLS.config_function(self.config_path, ["Combinatorial_Encoder"]))
                self.combinatorial_factor:int  = int(self.config["Combinatorial_Encoder"]["combinatorial_factor"])
                if self.run_length % self.combinatorial_factor != 0: raise Exception("run_time must be a multiple of combinatorial_factor -> run_time % combinatorial_factor == 0: run_time =",self.run_length, "combinatorial_factor =", self.combinatorial_factor)
                self.combinatorial_combinaison_size:float = float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_size"])
                self.combinatorial_combinaison_size_max:int = int(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_size_max"])
                self.combinatorial_combinaison_noise:float = float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_noise"])
                self.combinatorial_combinaison_noise_decay:float = float(self.config["Combinatorial_Encoder"]["combinatorial_combinaison_noise_decay"])
                if self.run_length > 50:
                    self.combinatorial_combinaisons:int = min(self.combinatorial_combinaison_size, self.combinatorial_combinaison_size_max)
                else:
                    self.combinatorial_combinaisons = min((2**(self.run_length//self.combinatorial_factor)), self.combinatorial_combinaison_size_max)
                self.combinatorial_combinaisons:int = min(self.combinatorial_combinaisons, self.combinatorial_combinaison_size)
                self.encoder_record_text:str = "combinatorial_factor-" + str(self.combinatorial_factor) + "_combinaison-" + str(self.combinatorial_combinaisons) + "_noise-" + str(self.combinatorial_combinaison_noise) + "_noise_decay-" + str(self.combinatorial_combinaison_noise_decay)
        
        if self.network_type == "ANN":
            self.is_bias = TOOLS.is_config_section(self.config_path, "bias_neuron_parameter")
            self.is_layer_normalization = True if self.config["Genome_NN"]["is_layer_normalization"] == "True" else False

        # 2 - Times parameters
        self.run_nb:int = 0
        self.generation:int = 0
        self.solved_at_generation:int = -1
        self.generations_times_list:List[float] = []


        self.is_parallele:bool = True if cpu > 1 else False
        if self.is_parallele == True:
            # 0 - Initialize Ray
            self.cpu:int = cpu if cpu <= os.cpu_count() else os.cpu_count() # give the number of logical cpu cores
            print("Parallel mode, CPU:", self.cpu)
            if ray.is_initialized() == True: # if ray is already initialized, shutdown and reinit
                ray.shutdown()
            ray.init(num_cpus=os.cpu_count(), include_dashboard=False) # number of cpu cores to use for the simulation
        else:
            self.cpu:int = 1
            print("Sequential mode CPU: 1")

    def __get_archives_names(self, config_path:str) -> List[str]:
        config:Dict[str, Dict[str, Any]] = TOOLS.config_function_all(config_path)
        self.archives_names:List[str] = []
        for section in config.keys():
            if section.startswith("Archive"):
                self.archives_names.append(section)
    
    def __update_archives(self, population:Population) -> None:
        if self.is_Archive == True and self.generation > 0:
            if self.archives == None:
                self.archives:List[Archive] = []
                for archive_name in self.archives_names:
                    self.archives.append(Archive(
                                            name=archive_name, 
                                            algorithm_name=self.algorithm_name, 
                                            config_path_file=self.config_path, 
                                            genome_builder_function=None, 
                                            nb_generation=self.nb_generations, 
                                            reproduction_size=self.algorithm.pop_size, 
                                            folder_path=self.record.folder_run_path if self.is_record == True else "./tpm_archive/"
                                            ))
            for archive in self.archives:
                archive.update(population)

    def __build_config_cache(self, config_path:str, config_cache_path:str) -> None:
        config_name = config_path.split("/")[-1]
        if not os.path.exists(config_cache_path):
            os.makedirs(config_cache_path)
        name = self.__init_file(config_cache_path, config_name)
        shutil.copyfile(config_path, config_cache_path + name)
        return config_cache_path + name
 
    def __init_file(self, config_cache_path, file_name:str) -> str:
        if os.path.exists(config_cache_path + file_name):
            i = 1
            while os.path.exists(config_cache_path + file_name + "_" + str(i)):
                i += 1
            return file_name + "_" + str(i) + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        else:
            return file_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")


    def init_algorithm(self, name:str, algorithm_builder:Algorithm, config_path:str) -> None:
        self.config_path_algorithm:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")
        self.algorithm_builder:Algorithm = algorithm_builder
        self.algorithm_name_init:str = name
        if name == "MAPELITE" or name == "NSLC":
            self.algorithm:Algorithm = self.algorithm_builder(self.config_path_algorithm, self.nb_generations, name, "test_results/")
        else:
            self.algorithm:Algorithm = self.algorithm_builder(self.config_path_algorithm, name)
        self.algorithm_name:str = self.config["NEURO_EVOLUTION"]["algo_name"]
    
    def re_init_algorithm(self, run_nb:int) -> Algorithm:
        if self.algorithm_name_init == "MAPELITE" or self.algorithm_name_init == "NSLC":
            self.algorithm:Algorithm = self.algorithm_builder(self.config_path_algorithm, self.nb_generations, self.algorithm_name_init, "test_results/")
        else:
            self.algorithm:Algorithm = self.algorithm_builder(self.config_path_algorithm, self.algorithm_name_init)

        if self.problem_type == "RL":
            self.obs_max:np.ndarray = None
            self.obs_min:np.ndarray = None
            self.seeds_RL:np.ndarray = self.seeds_from_param[run_nb]
            if self.is_parallele == True:
                [self.problems_ray[i].reset.remote(self.environment_builder, self.config_path_problem, self.seeds_RL, self.nb_episode) for i in range(self.cpu)]
                self.problem.reset(self.environment_builder, self.config_path_problem, self.seeds_RL, self.nb_episode)
            else:
                self.problem.reset(self.environment_builder, self.config_path_problem, self.seeds_RL, self.nb_episode)
        if self.record != None:
            self.record.best_genome = None

        return self.algorithm

    def init_problem_SL(self, problem_builder:Problem, config_path:str, name:str, features:np.ndarray, labels:np.ndarray) -> None:
        self.config_path_problem:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")
        self.problem_type:str = "SL"
        self.environment_name:str = name
        self.seeds_RL:np.ndarray = None
        genome_input_size:int = int(self.config["Genome_NN"]["inputs"])
        genome_output_size:int = int(self.config["Genome_NN"]["outputs"])
        if genome_input_size != len(features[0]): raise Exception("Error: input size is not the same as the data set, genome input size = " + str(genome_input_size), "data set size =" + str(len(features[0])))
        if genome_output_size != len(np.unique(labels)): raise Exception("Error: output size is not the same as the data set, genome output size = " + str(genome_output_size), "data set size =" + str(len(np.unique(labels))))

        if self.is_parallele == True:
            problem_builder_ray:Problem = ray.remote(problem_builder)
            features_id = ray.put(features)
            labels_id = ray.put(labels)
            self.problems_ray = [problem_builder_ray.remote(self.config_path_problem, features_id, labels_id, self.nb_generations) for _ in range(self.cpu)]
            print("Problem Ray:", self.problems_ray)
        else:
            self.problem:Problem = problem_builder(self.config_path_problem, features, labels, self.nb_generations)
            print("Problem:", self.problem)

    def init_problem_RL(self, problem_builder:Problem, config_path:str, environment, nb_episode:int, seeds:np.ndarray, render:bool=False) -> None:
        self.config_path_problem:str = self.__build_config_cache(config_path, os.getcwd() + "/config_cache/")
        self.environment_builder = environment
        self.environment_name:str = environment.name
        self.seeds_from_param:np.ndarray = seeds
        self.seeds_RL:np.ndarray = None
        self.nb_episode:int = nb_episode
        self.problem_type:str = "RL"
        if self.network_type == "SNN":
            self.is_online:bool = True if self.config["Runner_Info"]["online"] == "True" else False
        self.obs_max:np.ndarray = None
        self.obs_min:np.ndarray = None
        self.render:bool = render
        if self.is_parallele == True:
            problem_builder_ray:Problem = ray.remote(problem_builder)
            self.problems_ray = [problem_builder_ray.remote(self.environment_builder, self.config_path_problem, seeds, nb_episode, cpu_id=i, cpu_affinity=self.cpu_affinity) for i in range(self.cpu)]
            self.problem:Problem = problem_builder(self.environment_builder, self.config_path_problem, seeds, nb_episode)
            print("Problems Ray:", len(self.problems_ray))
        else:
            self.problem:Problem = problem_builder(self.environment_builder, self.config_path_problem, seeds, nb_episode)
            print("Problem Sequential: 1")


    def run(self):
        if self.is_parallele == True:
            self.run_parallele()
        else:
            self.run_sequential()

    def run_sequential(self):
        self.run_nb:int = 0

        for self.run_nb in range(self.nb_runs):
            self.algorithm = self.re_init_algorithm(run_nb=self.run_nb)
            population:Population = Population(get_new_population_id(), self.config_path)
            population.population = {}
            self.prev_obs_max:np.ndarray = None
            self.prev_obs_min:np.ndarray = None
            for self.generation in range(self.nb_generations):
                self.generation_time:float = time.time()
                    
                # Update Parameters
                population:Population = self.algorithm.run(population)

                # Run on Problem
                self.print_info_run()
                if self.problem_type == "RL":
                    self.prev_obs_max:np.ndarray = self.obs_max
                    self.prev_obs_min:np.ndarray = self.obs_min
                    population, self.obs_max, self.obs_min = self.problem.run(population, self.run_nb, self.generation, seeds=self.seeds_RL)
                    self.__set_population_obs_max_min(population, self.prev_obs_max, self.prev_obs_min)
                else:
                    population = self.problem.run(population, self.run_nb, self.generation, seed=None)

                # Record
                self.__update_records(population)

                # Archive
                self.__update_archives(population)

                # Update times info
                self.__update_info_run_and_print_stats(population)

        # 8 - Run render
        self.__render(population)


    def run_parallele(self):
        self.run_nb:int = 0
        pop_size:int = self.algorithm.pop_size
        if pop_size == None: raise ValueError("pop_size must be set in the algorithm")
        populations_cpu:List[Population] = [Population(get_new_population_id(), self.config_path) for _ in range(self.cpu)]

        for self.run_nb in range(self.nb_runs):
            self.algorithm = self.re_init_algorithm(run_nb=self.run_nb)
            population:Population = Population(get_new_population_id(), self.config_path)
            self.prev_obs_max:np.ndarray = None
            self.prev_obs_min:np.ndarray = None
            for self.generation in range(self.nb_generations):
                self.generation_time:float = time.time()
                    
                # 1 - Update population parameters
                population = self.algorithm.run(population)
                
                # 2 - Check and Split population for each cpu
                pop_size:int = len(population.population)
                if pop_size < self.cpu: raise ValueError("pop_size ("+ str(pop_size)+") must be equal or greater than cpu number (" + str(self.cpu) + ")")
                indexes:List[List[int]] = TOOLS.split(list(population.population.keys()), self.cpu)

                # 3 - Regroup cpu populations in a list of populations
                self.__set_populations_cpu(population, populations_cpu, indexes)

                # 4 - Run on Problem
                self.print_info_run()
                if self.generation != 0 and self.problem_type == "RL":
                    self.prev_obs_max:np.ndarray = self.obs_max
                    self.prev_obs_min:np.ndarray = self.obs_min
                    population_ids = [self.problems_ray[i].run.remote(populations_cpu[i], self.run_nb, self.generation, seeds=self.seeds_RL, indexes=None, obs_max=self.obs_max, obs_min=self.obs_min) for i in range(self.cpu)]
                else:
                    population_ids = [self.problems_ray[i].run.remote(populations_cpu[i], self.run_nb, self.generation, indexes=None) for i in range(self.cpu)]
                
                for i in range(self.cpu): populations_cpu[i].population = {} #new (for free memory)
                population.population = {} # new (for free memory)

                results = ray.get(population_ids)
                
                
                # 5 - Regroup population - Normal way
                if self.problem_type == "RL":
                    population_list, observations_max, observations_min = map(list, zip(*results))
                    population.update(population_list)
                    self.__set_population_obs_max_min(population, self.prev_obs_max, self.prev_obs_min)
                    self.obs_max, self.obs_min = self.__update_obersevation_stats(np.array(observations_max), np.array(observations_min))
                else:
                    population.update(results)


                # 6 - Record
                self.__update_records(population)

                # Archive
                self.__update_archives(population)

                # 7 - Update times info + print stats
                self.__update_info_run_and_print_stats(population)
                                
        ray.shutdown()
        # 8 - Run render
        self.__render(population)


    def run_runder_RL(self, best_genome:Genome, obs_max:np.ndarray, obs_min:np.ndarray) -> None:
        best_seed_index:int = 0
        score:float = None

        for i in range(len(self.seeds_RL)):
            if score == None: 
                score = best_genome.fitness.extra_info[i]
            if self.optimization_type == "maximize" and score < best_genome.fitness.extra_info[i]:
                score = best_genome.fitness.extra_info[i]
                best_seed_index = i
            elif self.optimization_type == "minimize" and score > best_genome.fitness.extra_info[i]:
                score = best_genome.fitness.extra_info[i]
                best_seed_index = i
            elif self.optimization_type == "closest_to_zero" and abs(score) > abs(best_genome.fitness.extra_info[i]):
                score = best_genome.fitness.extra_info[i]
                best_seed_index = i
            
            print("Seed:", self.seeds_RL[i], "Score:", best_genome.fitness.extra_info[i])
        print("Best Seed:", self.seeds_RL[best_seed_index], "Best Score:", score)
        self.problem.run_render(best_genome, self.environment_builder.get_env(render=True), self.seeds_RL[best_seed_index], obs_max=obs_max, obs_min=obs_min)

    
    def __set_populations_cpu(self, population:Population, populations_cpu:List[Population], indexes:List[List[int]]) -> None:
        for i in range(self.cpu):
            populations_cpu[i].population = {key:population.population[key] for key in indexes[i]}

    def __set_population_obs_max_min(self, population:Population, obs_max:np.ndarray, obs_min:np.ndarray) -> None:
        for genome in population.population.values():
            genome.info["obs_max"] = obs_max
            genome.info["obs_min"] = obs_min

    def __update_obersevation_stats(self, observation_max:np.ndarray, observation_min:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.max(observation_max, axis=0), np.min(observation_min, axis=0)

    def __get_archives_names(self, config_path:str) -> List[str]:
        config:Dict[str, Dict[str, Any]] = TOOLS.config_function_all(config_path)
        self.archives_names:List[str] = []
        for section in config.keys():
            if section.startswith("Archive"):
                self.archives_names.append(section)
    
    def __update_archives(self, population:Population) -> None:
        if self.is_Archive == True and self.generation > 0:
            if self.archives == None:
                self.archives:List[Archive] = []
                for archive_name in self.archives_names:
                    self.archives.append(Archive(
                                            name=archive_name, 
                                            algorithm_name=self.algorithm_name, 
                                            config_path_file=self.config_path, 
                                            genome_builder_function=None, 
                                            nb_generation=self.nb_generations, 
                                            reproduction_size=self.algorithm.pop_size, 
                                            folder_path=self.record.folder_run_path if self.is_record == True else "./tpm_archive/"
                                            ))
            for archive in self.archives:
                archive.update(population)

    def __update_records(self, population:Population) -> None:
        if self.is_record == True and self.generation > 0:
            if self.record == None:
                if self.problem_type == "RL":
                    if self.network_type == "SNN":
                        self.environment_name:str =  self.algorithm_name + "_" + self.encoder_record_text + "_" + self.decoder_record_text + "_" + self.problem_type + "_" + self.network_type + "_" + self.environment_name + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                    else:
                        self.environment_name:str =  self.algorithm_name + "_" +  self.problem_type + "_" + self.network_type + "_" + self.environment_name + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                else:
                    if self.network_type == "SNN":
                        self.environment_name:str =  self.algorithm_name + "_" + self.encoder_record_text + "_" + self.decoder_record_text + "_" + self.problem_type + "_" + self.network_type + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                    else:
                        self.environment_name:str =  self.algorithm_name + "_" + self.problem_type + "_" + self.network_type + "-pop-" + str(len(population.population)) + "-neuron-" + str(self.nb_hidden_neurons)
                self.record:Record = Record(self.config_path, self.environment_name)
            if self.algorithm_name == "MAPELITE" or self.algorithm_name == "NSLC":
                self.record.save_info(self.algorithm.population_best, self.run_nb)
            else:
                self.record.save_info(population, self.run_nb)

    def __update_info_run_and_print_stats(self, population:Population) -> None:
        self.generation += 1
        self.generation_time = time.time() - self.generation_time
        if len(self.generations_times_list) <= 7: self.generations_times_list.append(self.generation_time)
        else: self.generations_times_list.pop(0); self.generations_times_list.append(self.generation_time)

        if self.problem_type == "RL":
            self.__print_stats_RL(population)
        else:
            self.__print_stats_SL(population)
        if self.network_type == "SNN" and self.encoder == "combinatorial":
            self.combinatorial_combinaison_noise = self.combinatorial_combinaison_noise * self.combinatorial_combinaison_noise_decay

    def __render(self, population:Population) -> None:
        if self.problem_type == "RL" and self.render == True:
            # note -> on ne peut faire le meilleur genome ever car les obs_max et obs_min ne sont pas les memes à la fin de la run
            if self.algorithm_name == "MAPELITE" or self.algorithm_name == "NSLC":
                print("QD RENDER")
                QD_population:Population = self.algorithm.population_best
                QD_population.update_info()
                QD_best:Genome = QD_population.best_genome
                print("QD_best:", QD_best, "id",QD_best.id)
                print("QD_best observation:", QD_best.info["obs_max"], QD_best.info["obs_min"])
                self.run_runder_RL(QD_best, QD_best.info["obs_max"], QD_best.info["obs_min"])
            else:
                self.run_runder_RL(population.best_genome, self.prev_obs_max, self.prev_obs_min)

    def print_info_run(self) -> None:
        if self.network_type == "SNN":
            # GENERAL INFO
            print(
            "network_type:", self.network_type,
            "optimization_type:", self.optimization_type,
            "env_name:", self.environment_name,
            "record:", self.is_record,
            "archive:", self.is_Archive,
            "seeds:", self.seeds_RL,
            "cpu:", self.cpu
            )
            combinatorial_print:str = "combinatorial_noise: " + str(round(self.combinatorial_combinaison_noise, 3)) + " combinatorial_combinaisons: " + str(self.combinatorial_combinaisons) if self.is_combinatorial_encoding == True else ""
            voltage_print:str = "is_output_threshold: " + str(self.is_output_threshold) + " voltage_min: " + self.voltage_min + " voltage_max: " + self.voltage_max if self.is_voltage_encoding == True else ""
            if self.problem_type == "RL":
                print("encoder:", self.encoder, "decoder:", self.decoder, "record:", self.record_run, "decay:", self.decay_method, "is_output_threshold:", voltage_print, combinatorial_print)
                print("run_lenght:", self.run_length, "run_with_margin:", self.run_with_margin, "dt:", self.dt, "is_refractory:", self.is_refractory, "is_delay:", self.is_delay, "is_online:", self.is_online, "max_possible_spikes:", self.max_possible_spikes)
            else:
                print("run_lenght:", self.run_length, "run_with_margin:", self.run_with_margin, "dt:", self.dt, "is_refractory:", self.is_refractory, "is_delay:", self.is_delay, "encoder:", self.encoder, "decoder:", self.decoder, "record", self.record_run, "decay:", self.decay_method, voltage_print, combinatorial_print)

        if self.network_type == "ANN":
            is_bias:str = "bias: True" if self.is_bias == True else "bias: False"
            is_layer_normalization:str = "layer_norm: True" if self.is_layer_normalization == True else "layer_norm: False"
            # GENERAL INFO
            print(
            "network_type:", self.network_type,
            is_bias,
            is_layer_normalization,
            "optimization_type:", self.optimization_type,
            "env_name:", self.environment_name,
            "record:", self.is_record,
            "archive:", self.is_Archive,
            "seeds:", self.seeds_RL,
            "cpu:", self.cpu
            )

    def print_all_fitness(self, population:Population) -> None:
        fitnesses = []
        for genome in population.population.values():
            fitnesses.append(genome.fitness.score)
        fitnesses.sort(reverse=True)
        print("Fitnesses:", fitnesses)

    def __get_info_stats_by_alogrithm_SL(self, population:Population):
        population.update_info()
        best:Genome = population.best_genome
        stats:List[List[str, int, Tuple[int, float, int, int], float, int]] = []
        algorithm = self.algorithm # new
        algorithm.population_manager.update_info()
        name:str = algorithm.name
        pop_size:int = len(population.population)
        best:Genome = algorithm.population_manager.best_genome
        best_info:Tuple[int, float, int, int] = (best.id, round(best.fitness.score, 3), len(best.nn.hiddens["neurons_indexes_active"]), len(best.nn.synapses_actives_indexes[0]))
        avg_fitness:float = round(algorithm.population_manager.fitness.mean, 3)
        stats.append([name, pop_size, best_info, avg_fitness])
        return stats
    
    def __print_stats_SL(self, population:Population):
        if self.verbose == False: return
        minutes:float = (np.mean(self.generations_times_list)*(self.nb_generations - self.generation)) // 60
        seconds:float = (self.generation_time*(self.nb_generations - self.generation)) % 60
        time_left:str = str(int(minutes)) + "." + str(int(seconds))
        print("NEURO_EVO Run:",self.run_nb+1,"/", self.nb_runs, 
        "Generation:", self.generation, "/", self.nb_generations, 
        "pop_size:", len(population.population), 
        "time:", round(self.generation_time, 3), "s",
        "est_time: " + time_left +  " min",
        )
        t = time.localtime()
        currrent_time = time.strftime("%H:%M:%S", t)
        print("----------------------------------------------------------------------------------------------------------------------> local time: [" + str(currrent_time) + "]")
        titles = [["Algorithm", "Pop Size" ,"Best(id, fit, neur, syn)", "Avg fitness (pop)"]]
        titles.extend(self.__get_info_stats_by_alogrithm_SL(population))
        col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for row in titles:
            print("".join(str(word).ljust(col_width) for word in row))
        print("\n")

    def __get_info_stats_by_alogrithm_RL(self, population:Population):
        population.update_info()
        best:Genome = population.best_genome
        stats:List[List[str, int, Tuple[int, float, int, int], float, int]] = []
        algorithm = self.algorithm # new
        name:str = algorithm.name
        pop_size:int = len(population.population)
        best_episode:float = round(best.info["best_episode_raw_score"], 3)
        mean_episodes:float = round(best.info["mean_episode_raw_score"], 3)
        cum_episodes:float = round(best.fitness.score, 3)
        best_info:Tuple[int, float, float, float, int, int] = (best.id, best_episode, mean_episodes, cum_episodes, len(best.nn.hiddens["neurons_indexes_active"]), len(best.nn.synapses_actives_indexes[0]))
        avg_fitness:float = round(population.fitness.mean, 3)
        stats.append([name, pop_size, best_info, avg_fitness])
        return stats

    def __print_stats_RL(self, population:Population):
        if self.prev_obs_max is not None:
            print("observation_max:",np.round(self.prev_obs_max, 3))
            print("observation_min:", np.round(self.prev_obs_min, 3))
        if self.verbose == False: return
        minutes:float = (np.mean(self.generations_times_list)*(self.nb_generations - self.generation)) // 60
        seconds:float = (self.generation_time*(self.nb_generations - self.generation)) % 60
        time_left:str = str(int(minutes)) + "." + str(int(seconds))
        print("Run:",self.run_nb+1,"/", self.nb_runs, 
        "Generation:", self.generation, "/", self.nb_generations, 
        "pop_size:", len(population.population), 
        "time:", round(self.generation_time, 3), "s",
        "est_time: " + time_left +  " min"
        )
        t = time.localtime()
        currrent_time = time.strftime("%H:%M:%S", t)
        print("----------------------------------------------------------------------------------------------------------------------> local time: [" + str(currrent_time) + "]")
        titles = [["Algorithm", "Pop Size", "Best(id, fit, mean, cum, neur, syn)", "    Avg fitness (pop)"]]
        titles.extend(self.__get_info_stats_by_alogrithm_RL(population))
        # col_width = max(len(str(word)) for row in titles for word in row) + 2  # padding
        for j, row in enumerate(titles):
            # print("".join(str(word).ljust(col_width) for word in row))
            row_str:str = ""
            for i, word in enumerate(row):
                if i != 2:
                    row_str += str(word).ljust(15)
                if i == 2:
                    if j == 0:
                        row_str += str(word).ljust(50)
                    else:
                        row_str += "    " + str(word).ljust(50)
            print(row_str)
        print("\n")


    def check_snn_refractory_and_delay(self) -> None:
        if TOOLS.is_config_section(self.config_path, "refractory_neuron_parameter") == False:
            self.is_refractory:bool = False
        else:
            config_snn:Dict[str, Any] = TOOLS.config_function(self.config_path, ["refractory_neuron_parameter"])
            self.is_refractory:bool = np.any(np.array([
                float(config_snn["refractory_neuron_parameter"]["max"]),
                float(config_snn["refractory_neuron_parameter"]["min"])]) != 0.0)
        if TOOLS.is_config_section(self.config_path, "delay_synapse_parameter") == False:
            self.is_delay:bool = False
        else:
            config_snn:Dict[str, Any] = TOOLS.config_function(self.config_path, ["delay_synapse_parameter"])
            self.is_delay:bool = np.any(np.array([
                float(config_snn["delay_synapse_parameter"]["max"]),
                float(config_snn["delay_synapse_parameter"]["min"])]) != 0.0)
        
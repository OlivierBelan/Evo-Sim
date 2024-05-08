from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.NN import NN
import evo_simulator.TOOLS as TOOLS
from snn_api_cython import SNN
from typing import List, Dict, Any
import spikegen
import numpy as np
import torch


class Runner_Info:
    def __init__(self, config_path:str, nb_episode:int=0) -> None:
        self.config_dict:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Runner_Info", "Genome_NN"])
        self.config_path:str = config_path
        self.network_type:str = self.config_dict["Genome_NN"]["network_type"]
        self.runner_type:str = self.config_dict["Runner_Info"]["runner_type"]

        if self.network_type == "SNN":
            self.run_time:int = int(self.config_dict["Runner_Info"]["run_time"])
            self.run_time_margin:int = int(self.config_dict["Runner_Info"]["run_time_margin"])
            self.dt:float = float(self.config_dict["Runner_Info"]["dt"])
            self.neuron_reset:str = self.config_dict["Runner_Info"]["neuron_reset"]
            self.disable_output_threshold:bool = False
            self.decay_method:str = self.config_dict["Runner_Info"]["decay_method"]
            self.encoder:str = self.config_dict["Runner_Info"]["encoder"]
            self.decoder:str = self.config_dict["Runner_Info"]["decoder"]
            self.record:set[str] = set(self.config_dict["Runner_Info"]["record"].replace(",", " ").replace("  ", " ").split(" "))
            self.record_layer:List[str] = self.config_dict["Runner_Info"]["record_layer"].replace(",", " ").replace("  ", " ").split(" ")



            if self.decay_method not in ["lif", "beta"]: raise Exception("decay_method", self.decay_method, "not supported, please choose between 'lif' and 'beta'")
            if self.encoder not in ["poisson", "binomial", "exact", "rate", "combinatorial", "latency", "burst"]: raise Exception("encoding_method", self.encoder, "not supported, please choose between 'poisson', 'binomial', 'exact', 'rate', 'combinatorial', 'latency' and 'burst'")
            if self.decoder not in ["spike", "voltage", "augmented", "rate","max_spike", "coeff"]: raise Exception("decoding_method", self.decoding_method, "not supported, please choose between 'spike', 'voltage', 'augmented', 'rate' and 'max_spike'")
            for record_type in self.record:
                if record_type not in ["spike", "voltage", "augmented"]: raise Exception("record_decoding_method", record_type, "not supported, please choose between 'spike', 'voltage' or 'augmented'")            
            self.record_type:str = list(self.record)[0]
            if len(set(self.record_layer) - set(["input", "output", "hidden"])) != 0: raise Exception("record_layer", self.record_layer, "not supported, please choose between/or all 'input', 'output' and 'hidden'")

            if TOOLS.is_config_section(config_path, "delay_synapse_parameter") == False:
                self.is_delay = False
                self.delay_max:int = 0
            else:
                self.config_dict.update(TOOLS.config_function(config_path, ["delay_synapse_parameter"]))
                self.is_delay = True
                self.delay_max:int = float(self.config_dict["delay_synapse_parameter"]["max"])
                self.delay_max:int = np.ceil(self.delay_max).astype(int)

            self.is_refractory = TOOLS.is_config_section(config_path, "refractory_neuron_parameter")

            self.init_encoder(config_path)
            self.init_decoder(config_path)

        
        # used in Supervised
        if self.runner_type == "Supervised":
            self.batch_population:int = int(self.config_dict["Runner_Info"]["batch_population"])
            self.batch_features:int = int(self.config_dict["Runner_Info"]["batch_features"])
            self.batch_running:int = int(self.config_dict["Runner_Info"]["batch_running"])
            self.batch_running = self.batch_running if self.batch_running < self.batch_features else self.batch_features
        
        # used in Reinfocement
        if self.runner_type == "Reinforcement":
            if nb_episode == 0: raise Exception("nb_episode must be > 0")
            self.nb_episode:int = nb_episode
            if self.network_type == "SNN":
                self.online:bool = True if self.config_dict["Runner_Info"]["online"] == "True" else False

    def init_encoder(self, config_path:str) -> None:
        # ENCODERS: poisson, binomial, exact, rate, combinatorial, latency
        self.spike_rate:int = 0
        self.max_nb_spikes:int = 0
        self.reduce_noise:int = 0
        # for combinatorial init
        self.combinatorial_factor:int = 0
        self.combinatorial_combinaison_size:float = 0
        self.combinatorial_combinaison_size_max:int = 0
        self.combinatorial_combinaison_noise:float = 0
        self.combinatorial_combinaison_noise_decay:float = 0
        self.combinatorial_is_first_decay:bool = False
        self.combinatorial_roll:bool = False
        
        if self.encoder == "poisson":
            self.config_dict.update(TOOLS.config_function(config_path, ["Poisson_Encoder"]))
            self.spike_rate:int = int(self.config_dict["Poisson_Encoder"]["spike_rate"])
            self.max_nb_spikes:int = int(self.config_dict["Poisson_Encoder"]["max_nb_spikes"])
            self.spike_amplitude = int(self.config_dict["Poisson_Encoder"]["spike_amplitude"])

        elif self.encoder == "binomial":
            self.config_dict.update(TOOLS.config_function(config_path, ["Binomial_Encoder"]))
            self.reduce_noise:int = int(self.config_dict["Binomial_Encoder"]["reduce_noise"])
            self.max_nb_spikes:int = int(self.config_dict["Binomial_Encoder"]["max_nb_spikes"])
            self.spike_amplitude = int(self.config_dict["Binomial_Encoder"]["spike_amplitude"])
        
        elif self.encoder == "exact":
            self.config_dict.update(TOOLS.config_function(config_path, ["Exact_Encoder"]))
            self.max_nb_spikes:int = int(self.config_dict["Exact_Encoder"]["max_nb_spikes"])
            self.spike_amplitude = int(self.config_dict["Exact_Encoder"]["spike_amplitude"])
                
        elif self.encoder == "combinatorial":
            self.config_dict.update(TOOLS.config_function(config_path, ["Combinatorial_Encoder"]))
            self.combinatorial_factor:int = int(self.config_dict["Combinatorial_Encoder"]["combinatorial_factor"])
            self.combinatorial_combinaison_size:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_size"])
            self.combinatorial_combinaison_size_max:int = int(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_size_max"])
            self.combinatorial_combinaison_noise:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_noise"])
            self.combinatorial_combinaison_noise_decay:float = float(self.config_dict["Combinatorial_Encoder"]["combinatorial_combinaison_noise_decay"])
            self.combinatorial_is_first_decay:bool = True
            self.combinatorial_roll:bool = True if self.config_dict["Combinatorial_Encoder"]["combinatorial_roll"] == "True" else False
            if self.run_time % self.combinatorial_factor != 0: raise Exception("run_time must be a multiple of combinatorial_factor -> run_time % combinatorial_factor == 0: run_time =",self.run_time, "combinatorial_factor =", self.combinatorial_factor)
            self.spike_amplitude = int(self.config_dict["Combinatorial_Encoder"]["spike_amplitude"])
        elif self.encoder == "rate":
            self.config_dict.update(TOOLS.config_function(config_path, ["Rate_Encoder"]))
            self.spike_amplitude = int(self.config_dict["Rate_Encoder"]["spike_amplitude"])
        
        elif self.encoder == "latency":
            self.config_dict.update(TOOLS.config_function(config_path, ["Latency_Encoder"]))
            self.spike_amplitude = int(self.config_dict["Latency_Encoder"]["spike_amplitude"])
        else:
            raise Exception("Encoder", self.encoder, "not supported, please choose between 'poisson', 'binomial', 'exact', 'rate', 'combinatorial' and 'latency'")

    def init_decoder(self, config_path:str) -> None:
        # DECODERS: rate, voltage, augmented, max_spike, coeff
        if self.decoder == "rate":
            self.config_dict.update(TOOLS.config_function(config_path, ["Rate_Decoder"]))
            self.ratio_max_output_spike = np.clip(float(self.config_dict["Rate_Decoder"]["ratio_max_output_spike"]), 0.0, 1.0)
        
        elif self.decoder == "voltage":
            self.config_dict.update(TOOLS.config_function(config_path, ["Voltage_Decoder"]))
            self.disable_output_threshold:bool = True if self.config_dict["Voltage_Decoder"]["disable_output_threshold"] == "True" else False
            nb_outputs:int = int(self.config_dict["Genome_NN"]["outputs"])
            self.voltage_max:str = self.config_dict["Voltage_Decoder"]["voltage_max"]
            self.voltage_min:str = self.config_dict["Voltage_Decoder"]["voltage_min"]
            self.is_voltages_min_decoder:bool = True if self.voltage_min == "voltage" else False
            self.is_threshold_max_decoder:bool = True if self.voltage_max == "threshold" else False
            if self.is_threshold_max_decoder == False:
                try:
                    self.voltage_max:np.ndarray = np.full(nb_outputs, float(self.voltage_max))
                except:
                    raise Exception("voltage_max:", self.voltage_max, "not supported, please choose a float value or 'threshold'")
            if self.is_voltages_min_decoder == False:
                try:
                    self.voltage_min:np.ndarray = np.full(nb_outputs, float(self.voltage_min))
                except:
                    raise Exception("voltage_min:", self.voltage_min, "not supported, please choose a float value or 'voltage'")

        
        elif self.decoder == "augmented":
            self.config_dict.update(TOOLS.config_function(config_path, ["Augmented_Decoder"]))
            self.spike_max:int = int(self.config_dict["Augmented_Decoder"]["spike_max"])
            self.spike_distribution_run:int = int(self.config_dict["Augmented_Decoder"]["spike_distribution_run"])
            self.spike_distribution_importance:int = int(self.config_dict["Augmented_Decoder"]["spike_distribution_importance"])
            # positive, absolute, raw (positive and negative)
            self.spike_type:str = self.config_dict["Augmented_Decoder"]["spike_type"]
            # first_index, by_index, all, nothing 
            self.importance_type:str = self.config_dict["Augmented_Decoder"]["importance_type"]
            # ascending, descending
            self.linear_spike_importance_type:str = self.config_dict["Augmented_Decoder"]["linear_spike_importance_type"]
            self.output_multiplier:float = float(self.config_dict["Genome_NN"]["outputs_multiplicator"])

            if self.spike_type not in ["positive", "absolute", "raw"]: raise Exception("spike_type", self.spike_type, "not supported, please choose between 'positive', 'absolute' and 'raw'")
            if self.importance_type not in ["first_index", "by_index", "all", "nothing"]: raise Exception("importance_type", self.importance_type, "not supported, please choose between 'first_index', 'by_index', 'all' and 'nothing'")
            if self.linear_spike_importance_type not in ["ascending", "descending"]: raise Exception("linear_spike_importance_type", self.linear_spike_importance_type, "not supported, please choose between 'ascending' and 'descending'")
        
        elif self.decoder == "max_spike":
            pass
        
        elif self.decoder == "coeff":
            pass
        else:
            raise Exception("Decoder", self.decoder, "not supported, please choose between 'rate', 'voltage', 'augmented', 'max_spike' and 'coeff'")

class Runner(Runner_Info):
    def __init__(self, config_path:str, nb_episode:int=None) -> None:
        Runner_Info.__init__(self, config_path, nb_episode) # Get SNN Runner Info from config file

        # 1 - INIT runner type (Supervised Learning or Reinforcement Learning)
        if self.runner_type not in ["Supervised", "Reinforcement"]:
            raise Exception("Runner type", self.runner_type, "not supported, please choose between 'Supervised' and 'Reinforcement'")
        

        # 3 - INIT Variables
        self.snn_list_len:int = 0
        self.snns_cython_dict:Dict[int, SNN] = {}
        self.neurons_total_used:int = 0
        self.synapses_total_used:int = 0
        self.cython_init:bool = False

    def __init_cython_runner(self) -> None:
        if self.cython_init == True: return

        from snn_cython import SNN_cython

        if self.runner_type == "Supervised": # Supervised Learning Runner
            from runner_SL_cython import Runner_SL_cython
            self.__runner_cython = Runner_SL_cython()

        elif self.runner_type == "Reinforcement": # Reinforcement Learning Runner
            from runner_RL_cython import Runner_RL_cython
            self.__runner_cython = Runner_RL_cython()


        self.SNN_cython_class = SNN_cython
        self.cython_init:bool = True

        if "augmented" in self.record:
            self.__runner_cython.init_augmented_spikes_decoder(self.spike_distribution_run, self.spike_distribution_importance, self.importance_type, self.linear_spike_importance_type, self.spike_type)


    def init_networks(self, genomes:List[Genome_NN]) -> Dict[int, NN]:
        self.__init_cython_runner()
        self.snn_list_len = len(genomes)
        
        # 2 - Create snns_cython population
        self.snns_cython_dict:Dict[int, NN] = self.create_snns_population(genomes) # create snns_cython population

        # 3 -  Init networks
        if self.runner_type == "Supervised":
            self.__runner_cython.init(list(self.snns_cython_dict.values()), run_time=self.run_time, run_time_margin=self.run_time_margin, batch_features=self.batch_features, batch_running=self.batch_running, batch_population=self.batch_population, neuron_reset=self.neuron_reset, record_layer=self.record_layer, disable_output_threshold=self.disable_output_threshold, decay_method=self.decay_method, is_delay=self.is_delay, is_refractory=self.is_refractory, record_decoding_method=self.record)
        elif self.runner_type == "Reinforcement":
            self.__runner_cython.init(list(self.snns_cython_dict.values()), run_time=self.run_time, run_time_margin=self.run_time_margin, dt=self.dt, nb_episode=self.nb_episode, online=self.online, neuron_reset=self.neuron_reset, record_layer=self.record_layer, disable_output_threshold=self.disable_output_threshold, decay_method=self.decay_method, is_delay=self.is_delay, is_refractory=self.is_refractory, delay_max=self.delay_max, record_decoding_method=self.record)

        if self.encoder == "combinatorial" and self.combinatorial_is_first_decay == False:
            self.combinatorial_combinaison_noise = self.combinatorial_combinaison_noise * self.combinatorial_combinaison_noise_decay
        self.combinatorial_is_first_decay = False

        return self.snns_cython_dict


    def run(self, features:List[np.ndarray] | np.ndarray = None) -> Dict[str, Dict[int, np.ndarray]]:
        records:Dict[str, Dict[int, np.ndarray]] = {}
        
        # 1 - Run networks Supervised
        if (self.runner_type == "Supervised"):
            self.__runner_cython.run()
        
        # 1 - Run networks Reinforcement
        elif (self.runner_type == "Reinforcement" and self.snn_list_len == len(features[0])): # Check if nb of snns == nb of features
            features = self.check_encoder(features)
            self.__runner_cython.run(self.encoder, features, self.spike_rate, self.spike_amplitude, self.max_nb_spikes, self.reduce_noise, self.combinatorial_factor, self.combinatorial_combinaison_size, self.combinatorial_combinaison_size_max, self.combinatorial_combinaison_noise, self.combinatorial_roll)

        else:
            print("runner type", self.runner_type)
            print("snn_list_len", self.snn_list_len, "features", len(features[0]))
            print("features", features, "shape", features[0].shape)
            raise ValueError("Runner type", self.runner_type, "not supported, please choose between 'Supervised' and 'Reinforcement'")

        if "spike"     in self.record: records["spike"] = self.__runner_cython.get_record_spikes()
        if "voltage"   in self.record: records["voltage"] = self.__runner_cython.get_record_voltages()
        if "augmented" in self.record: records["augmented"] = self.__runner_cython.get_record_augmented_spikes()
        # print("records", records)
        # exit()

        return records

    def create_snns_population(self, genomes:List[Genome_NN]) -> Dict[int, NN]:
        self.snn_list_len = len(genomes)

        # 1 - Get neuron_actives_indexes_pop and synapses_actives_indexes_pop
        neuron_actives_indexes_pop:np.ndarray = None
        synapses_actives_indexes_pop:np.ndarray = None
        for genome in genomes:
            if neuron_actives_indexes_pop is None:
                neuron_actives_indexes_pop:np.ndarray = np.zeros(genome.nn.nb_neurons, dtype=np.int32)
                synapses_actives_indexes_pop:np.ndarray = np.zeros((genome.nn.nb_neurons, genome.nn.nb_neurons), dtype=np.int32)
            neuron_actives_indexes_pop[genome.nn.neuron_actives_indexes] = 1
            # print("genome.snn.neuron_actives_indexes", genome.snn.neuron_actives_indexes)
            # print("genome.snn.synapses_actives_indexes", genome.snn.synapses_actives_indexes)
            # print("genome.snn.hidden_neurons_indexes_active", genome.snn.hiddens["neurons_indexes_active"])
            synapses_actives_indexes_pop[genome.nn.synapses_actives_indexes[0], genome.nn.synapses_actives_indexes[1]] = 1
        neuron_actives_indexes_pop = np.where(neuron_actives_indexes_pop == 1)[0].astype(np.int32)
        synapses_actives_indexes_pop = np.array(np.where(synapses_actives_indexes_pop == 1), dtype=np.int32)

        # Just in order to get useful info
        self.neurons_total_used = neuron_actives_indexes_pop.shape[0]
        self.synapses_total_used = synapses_actives_indexes_pop.shape[1]

        # 2 - Create cython SNNs
        snns_dict:Dict[int, SNN] = {}
        for genome in genomes:
            snn_python:NN = genome.nn
            snn_cython = self.SNN_cython_class(genome.id) # Create SNN_cython object
            snn_cython.init_network(snn_python.parameters, 
                                    snn_python.nb_inputs,
                                    snn_python.nb_outputs,
                                    len(snn_python.hiddens["neurons_indexes_active"]),
                                    snn_python.inputs["neurons_indexes"],
                                    snn_python.outputs["neurons_indexes"], 
                                    snn_python.hiddens["neurons_indexes_active"],
                                    # np.array(snn_python.neuron_actives_indexes, dtype=np.int32),
                                    # np.array(snn_python.synapses_actives_indexes, dtype=np.int32),
                                    neuron_actives_indexes_pop,
                                    # synapses_actives_indexes_pop,
                                    np.array(snn_python.synapses_unactives_weight_indexes), # unactives synapses indexes
                                    np.where(~snn_python.neurons_status)[0].astype(np.int32), # unactives neurons indexes
                                    self.is_delay,
                                    self.is_refractory,
                                    )
            snns_dict[genome.id] = snn_cython
        return snns_dict


    def init_inputs_networks(self, features:np.ndarray) -> None:
        if self.encoder == "poisson":
            self.poisson_encoder(features)
        elif self.encoder == "binomial":
            self.binomial_encoder(features)
        elif self.encoder == "exact":
            self.exact_encoder(features)
        elif self.encoder == "rate":
            self.rate_encoder(features)
        elif self.encoder == "combinatorial":
            self.combinatorial_encoder(features)
        elif self.encoder == "latency":
            self.latency_encoder(features)
        # elif self.encoder == "burst":
        #     self.burst_encoder(features)
        else:
            raise Exception("Encoder", self.encoder, "not supported, please choose between 'poisson', 'binomial', 'exact', 'rate', 'combinatorial' and 'latency'")

    def poisson_encoder(self, features:np.ndarray | List[np.ndarray]) -> None:
        if self.runner_type == "Supervised":
            self.__runner_cython.poisson_encoder(features, self.spike_rate, self.spike_amplitude, self.max_nb_spikes)

        elif self.runner_type == "Reinforcement":
            print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
            exit()

    def binomial_encoder(self, features:np.ndarray | List[np.ndarray]) -> None:
        if self.runner_type == "Supervised":
            self.__runner_cython.binomial_encoder(features, self.spike_amplitude, self.max_nb_spikes, self.reduce_noise)   

        elif self.runner_type == "Reinforcement":
            print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
            exit()

    def exact_encoder(self, features:np.ndarray | List[np.ndarray]) -> None:
        if self.runner_type == "Supervised":
            self.__runner_cython.exact_encoder(features, self.spike_amplitude, self.max_nb_spikes)
        
        elif self.runner_type == "Reinforcement":
            print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
            exit()

    def rate_encoder(self, features:np.ndarray | List[np.ndarray]) -> None:
        if self.runner_type == "Supervised":
            self.__runner_cython.rate_encoder(features, self.spike_amplitude)

        elif self.runner_type == "Reinforcement":
            print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
            exit()

    def latency_encoder(self, features:np.ndarray) -> None:
        if self.runner_type == "Supervised":
            features = spikegen.latency(torch.tensor(features), num_steps=self.run_time, tau=5, threshold=0.01, clip=True, normalize=True, linear=True).numpy().T
            features = np.transpose(features, (1, 0, 2))
            self.__runner_cython.raw_encoder(features, self.spike_amplitude)
        
        elif self.runner_type == "Reinforcement":
            print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
            exit()

    def combinatorial_encoder(self, features:np.ndarray) -> None:
        if self.runner_type == "Supervised":
            self.__runner_cython.combinatorial_encoder(features, self.combinatorial_factor, self.combinatorial_combinaison_size, self.combinatorial_combinaison_size_max, self.combinatorial_combinaison_noise, self.combinatorial_roll, self.spike_amplitude)
        
        elif self.runner_type == "Reinforcement":
            print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
            exit()

    # def burst_encoder(self, features:np.ndarray) -> None:
    #     if self.runner_type == "Supervised":
    #         self.__runner_cython.burst_encoder(features, self.spike_amplitude, self.max_nb_spikes)
        
    #     elif self.runner_type == "Reinforcement":
    #         print("FROM runner_api.cython.py: HAS TO CHECK THIS FUNCTION FOR RL FEATURES")
    #         exit()

    def check_encoder(self, features:np.ndarray) -> np.ndarray:
        if self.encoder == "latency":
            features = spikegen.latency(torch.tensor(features), num_steps=self.run_time, tau=5, threshold=0.01, clip=True, normalize=True, linear=True).numpy().T
            features = np.transpose(features, (2, 1, 0, 3)).astype(np.float32)
        return features
            
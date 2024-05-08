from typing import Dict, Any
import psutil
class SNN_Manager:
    def __init__(self, config:Dict[str,str]) -> None:
        self.cpu:int = min(psutil.cpu_count(logical=True), int(config["cpu"]))
        self.dt= float(config["dt"])
        self.run_time= float(config["run_time"])
        self.nb_episodes = int(config["nb_episodes"]) # used in REINFORCEMENT
        self.batch_features= int(config["batch_features"])
        self.batch_population= int(config["batch_population"])
        self.extra:Dict[str, Any] = {}

        # POISSON, BINOMIAL and EXACT parameters
        self.inputs_type:str = config["inputs_type"] # poisson, binomial or exact
        self.spike_amplitude:float = float(config["spike_amplitude"]) # used in POISSON, BINOMIAL and EXACT
        self.max_nb_spikes:int = int(config["max_nb_spikes"])# used in POISSON, BINOMIAL and EXACT
        self.reduce_noise_loop:int = int(config["reduce_noise"]) # used in POISSON and BINOMIAL
        self.spike_rate:int = int(config["spike_rate"])# used in POISSON
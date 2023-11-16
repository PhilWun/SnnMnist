"""
Created on 15.12.2014

@author: Peter U. Diehl
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import brian2 as b2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from src.data_handler import (
    get_labeled_data,
    get_matrix_from_file,
    save_connections,
    save_theta,
)
from src.plotting import (
    plot_2d_input_weights,
    plot_performance,
    update_2d_input_weights,
    update_performance_plot,
    plot_results,
)


@dataclass
class NeuronModelHyperparameters:
    v_rest_e: b2.Quantity
    v_rest_i: b2.Quantity
    v_reset_e: b2.Quantity
    v_reset_i: b2.Quantity
    v_thresh_e: b2.Quantity
    v_thresh_i: b2.Quantity
    refrac_e: b2.Quantity
    refrac_i: b2.Quantity

    @staticmethod
    def get_default() -> "NeuronModelHyperparameters":
        return NeuronModelHyperparameters(
            v_rest_e=-65.0 * b2.mV,
            v_rest_i=-60.0 * b2.mV,
            v_reset_e=-65.0 * b2.mV,
            v_reset_i=-45.0 * b2.mV,
            v_thresh_e=-52.0 * b2.mV,
            v_thresh_i=-40.0 * b2.mV,
            refrac_e=5.0 * b2.ms,
            refrac_i=2.0 * b2.ms,
        )


@dataclass
class SynapseModelHyperparameters:
    tc_pre_ee: b2.Quantity
    tc_post_1_ee: b2.Quantity
    tc_post_2_ee: b2.Quantity
    nu_ee_pre: float
    nu_ee_post: float
    wmax_ee: float
    exp_ee_pre: float
    exp_ee_post: float
    STDP_offset: float
    tc_theta: b2.Quantity
    theta_plus_e: b2.Quantity
    offset: b2.Quantity

    @staticmethod
    def get_default() -> "SynapseModelHyperparameters":
        return SynapseModelHyperparameters(
            tc_pre_ee=20 * b2.ms,
            tc_post_1_ee=20 * b2.ms,
            tc_post_2_ee=40 * b2.ms,
            nu_ee_pre=0.0001,  # learning rate
            nu_ee_post=0.01,  # learning rate,
            wmax_ee=1.0,
            exp_ee_pre=0.2,
            exp_ee_post=0.2,
            STDP_offset=0.4,
            tc_theta=1e7 * b2.ms,
            theta_plus_e=0.05 * b2.mV,
            offset=20.0 * b2.mV,
        )


@dataclass
class ExperimentHyperparameters:
    test_mode: bool
    data_path: Path
    weight_path: Path
    num_examples: int
    use_testing_set: bool
    do_plot_performance: bool
    record_spikes: bool
    ee_stdp_on: bool
    update_interval: int
    ending: str
    n_input: int
    n_e: int
    n_i: int
    single_example_time: b2.Quantity
    resting_time: b2.Quantity
    runtime: b2.Quantity
    update_interval: int
    weight_update_interval: int
    save_connections_interval: int

    @staticmethod
    def get_default(test_mode: bool) -> "ExperimentHyperparameters":
        np.random.seed(0)
        data_path = Path(".")

        if test_mode:
            weight_path = data_path / "weights"
            num_examples = 10000 * 1
            use_testing_set = True
            do_plot_performance = False
            record_spikes = True
            ee_stdp_on = False
        else:
            weight_path = data_path / "random"
            num_examples = 600 * 3
            use_testing_set = False
            do_plot_performance = True

            if num_examples <= 60000:
                record_spikes = True
            else:
                record_spikes = True

            ee_stdp_on = True

        ending = ""
        n_input = 784
        n_e = 400
        n_i = n_e
        single_example_time = 0.35 * b2.second  #
        resting_time = 0.15 * b2.second
        runtime = num_examples * (single_example_time + resting_time)

        if num_examples <= 10000:
            update_interval = num_examples
            weight_update_interval = 20
        else:
            update_interval = 10000
            weight_update_interval = 100
        if num_examples <= 60000:
            save_connections_interval = 10000
        else:
            save_connections_interval = 10000
            update_interval = 10000

        return ExperimentHyperparameters(
            test_mode=test_mode,
            data_path=data_path,
            weight_path=weight_path,
            num_examples=num_examples,
            use_testing_set=use_testing_set,
            do_plot_performance=do_plot_performance,
            record_spikes=record_spikes,
            ee_stdp_on=ee_stdp_on,
            update_interval=update_interval,
            ending=ending,
            n_input=n_input,
            n_e=n_e,
            n_i=n_i,
            single_example_time=single_example_time,
            resting_time=resting_time,
            runtime=runtime,
            weight_update_interval=weight_update_interval,
            save_connections_interval=save_connections_interval,
        )


@dataclass
class NetworkArchitectureHyperparameters:
    weight: Dict[str, float]
    delay: Dict[str, Tuple[b2.Quantity, b2.Quantity]]
    input_population_names: List[str]
    population_names: List[str]
    input_connection_names: List[str]
    save_conns: List[str]
    input_conn_names: List[str]
    recurrent_conn_names: List[str]
    input_intensity: float
    start_input_intensity: float

    @staticmethod
    def get_default() -> "NetworkArchitectureHyperparameters":
        return NetworkArchitectureHyperparameters(
            weight={"ee_input": 78.0},
            delay={
                "ee_input": (0 * b2.ms, 10 * b2.ms),
                "ei_input": (0 * b2.ms, 5 * b2.ms),
            },
            input_population_names=["X"],
            population_names=["A"],
            input_connection_names=["XA"],
            save_conns=["XeAe"],
            input_conn_names=["ee_input"],
            recurrent_conn_names=["ei", "ie"],
            input_intensity=2.0,
            start_input_intensity=2.0,
        )


@dataclass
class ModelEquations:
    scr_e: str
    v_thresh_e_eqs: str
    v_thresh_i_eqs: str
    v_reset_i_eqs: str
    neuron_eqs_e: str
    neuron_eqs_i: str
    eqs_stdp_ee: str
    eqs_stdp_pre_ee: str
    eqs_stdp_post_ee: str

    @staticmethod
    def get_default(test_mode: bool) -> "ModelEquations":
        if test_mode:
            scr_e = "v = v_reset_e; timer = 0*ms"
        else:
            scr_e = "v = v_reset_e; theta += theta_plus_e; timer = 0*ms"

        v_thresh_e_eqs = "(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)"
        v_thresh_i_eqs = "v>v_thresh_i"
        v_reset_i_eqs = "v=v_reset_i"

        neuron_eqs_e = """
                       dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
                       I_synE = ge * nS *         -v                           : amp
                       I_synI = gi * nS * (-100.*mV-v)                          : amp
                       dge/dt = -ge/(1.0*ms)                                   : 1
                       dgi/dt = -gi/(2.0*ms)                                  : 1
                       """

        if test_mode:
            neuron_eqs_e += "\n  theta      :volt"
        else:
            neuron_eqs_e += "\n  dtheta/dt = -theta / (tc_theta)  : volt"
        neuron_eqs_e += "\n  dtimer/dt = 0.1  : second"

        neuron_eqs_i = """
                dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
                I_synE = ge * nS *         -v                           : amp
                I_synI = gi * nS * (-85.*mV-v)                          : amp
                dge/dt = -ge/(1.0*ms)                                   : 1
                dgi/dt = -gi/(2.0*ms)                                  : 1
                """
        eqs_stdp_ee = """
                        post2before                            : 1
                        dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                        dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                        dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
                    """
        eqs_stdp_pre_ee = "pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)"
        eqs_stdp_post_ee = "post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1."

        return ModelEquations(
            scr_e=scr_e,
            v_thresh_e_eqs=v_thresh_e_eqs,
            v_thresh_i_eqs=v_thresh_i_eqs,
            v_reset_i_eqs=v_reset_i_eqs,
            neuron_eqs_e=neuron_eqs_e,
            neuron_eqs_i=neuron_eqs_i,
            eqs_stdp_ee=eqs_stdp_ee,
            eqs_stdp_pre_ee=eqs_stdp_pre_ee,
            eqs_stdp_post_ee=eqs_stdp_post_ee,
        )


class Runner:
    def __init__(self):
        # ------------------------------------------------------------------------------
        # load MNIST
        # ------------------------------------------------------------------------------
        start = time.time()
        self.training_data = get_labeled_data("training")
        end = time.time()
        print("time needed to load training set:", end - start)

        start = time.time()
        self.testing_data = get_labeled_data("testing", b_train=False)
        end = time.time()
        print("time needed to load test set:", end - start)

        # ------------------------------------------------------------------------------
        # set parameters and equations
        # ------------------------------------------------------------------------------
        self.experiment_hyperparameters = ExperimentHyperparameters.get_default(
            test_mode=False
        )

        self.neuron_model_hyperparameters = NeuronModelHyperparameters.get_default()

        self.network_architecture_hyperparameters = (
            NetworkArchitectureHyperparameters.get_default()
        )

        self.synapse_model_hyperparameters = SynapseModelHyperparameters.get_default()

        self.model_equations = ModelEquations.get_default(
            self.experiment_hyperparameters.test_mode
        )

        b2.ion()
        self.fig_num = 1
        self.neuron_groups = {}
        self.input_groups: Dict[str, b2.PoissonGroup] = {}
        self.connections: Dict[str, b2.Synapses] = {}
        self.rate_monitors: Dict[str, b2.PopulationRateMonitor] = {}
        self.spike_monitors: Dict[str, b2.SpikeMonitor] = {}
        self.spike_counters: Dict[str, b2.SpikeMonitor] = {}
        self.result_monitor: np.ndarray = np.zeros(
            (
                self.experiment_hyperparameters.update_interval,
                self.experiment_hyperparameters.n_e,
            )
        )

        self.neuron_groups["e"] = b2.NeuronGroup(
            self.experiment_hyperparameters.n_e
            * len(self.network_architecture_hyperparameters.population_names),
            self.model_equations.neuron_eqs_e,
            threshold=self.model_equations.v_thresh_e_eqs,
            refractory=self.neuron_model_hyperparameters.refrac_e,
            reset=self.model_equations.scr_e,
            method="euler",
        )
        self.neuron_groups["i"] = b2.NeuronGroup(
            self.experiment_hyperparameters.n_i
            * len(self.network_architecture_hyperparameters.population_names),
            self.model_equations.neuron_eqs_i,
            threshold=self.model_equations.v_thresh_i_eqs,
            refractory=self.neuron_model_hyperparameters.refrac_i,
            reset=self.model_equations.v_reset_i_eqs,
            method="euler",
        )

        self.create_network_and_recurrent_connections()
        self.create_input_population_and_connection()

    def normalize_weights(self):
        for connName in self.connections:
            if connName[1] == "e" and connName[3] == "e":
                len_source = len(self.connections[connName].source)
                len_target = len(self.connections[connName].target)
                connection = np.zeros((len_source, len_target))
                connection[
                    self.connections[connName].i, self.connections[connName].j
                ] = self.connections[connName].w
                temp_conn = np.copy(connection)
                col_sums = np.sum(temp_conn, axis=0)
                col_factors = (
                    self.network_architecture_hyperparameters.weight["ee_input"]
                    / col_sums
                )

                for j in range(self.experiment_hyperparameters.n_e):  #
                    temp_conn[:, j] *= col_factors[j]

                self.connections[connName].w = temp_conn[
                    self.connections[connName].i, self.connections[connName].j
                ]

    @staticmethod
    def get_recognized_number_ranking(
        assignments: np.ndarray, spike_rates: np.ndarray
    ) -> np.ndarray:
        summed_rates = [0] * 10
        num_assignments = [0] * 10

        for i in range(10):
            num_assignments[i] = len(np.where(assignments == i)[0])
            if num_assignments[i] > 0:
                summed_rates[i] = (
                    np.sum(spike_rates[assignments == i]) / num_assignments[i]
                )

        return np.argsort(summed_rates)[::-1]

    def get_new_assignments(self, result_monitor: np.ndarray, input_numbers: List[int]):
        assignments: np.ndarray = np.zeros(self.experiment_hyperparameters.n_e)
        input_nums: np.ndarray = np.asarray(input_numbers)
        maximum_rate: List[float] = [0] * self.experiment_hyperparameters.n_e

        for j in range(10):
            num_assignments = len(np.where(input_nums == j)[0])
            assert num_assignments > 0

            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments

            for i in range(self.experiment_hyperparameters.n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j

        return assignments

    def create_network_and_recurrent_connections(self):
        # ------------------------------------------------------------------------------
        # create network population and recurrent connections
        # ------------------------------------------------------------------------------
        name: str
        subgroup_n: int

        for subgroup_n, name in enumerate(
            self.network_architecture_hyperparameters.population_names
        ):
            print("create neuron group", name)

            self.neuron_groups[name + "e"] = self.neuron_groups["e"][
                subgroup_n
                * self.experiment_hyperparameters.n_e : (subgroup_n + 1)
                * self.experiment_hyperparameters.n_e
            ]
            self.neuron_groups[name + "i"] = self.neuron_groups["i"][
                subgroup_n
                * self.experiment_hyperparameters.n_i : (subgroup_n + 1)
                * self.experiment_hyperparameters.n_e
            ]

            self.neuron_groups[name + "e"].v = (
                self.neuron_model_hyperparameters.v_rest_e - 40.0 * b2.mV
            )
            self.neuron_groups[name + "i"].v = (
                self.neuron_model_hyperparameters.v_rest_i - 40.0 * b2.mV
            )

            if (
                self.experiment_hyperparameters.test_mode
                or str(self.experiment_hyperparameters.weight_path)[-7:] == "weights"
            ):
                self.neuron_groups["e"].theta = (
                    np.load(
                        self.experiment_hyperparameters.weight_path
                        / (
                            "theta_"
                            + name
                            + self.experiment_hyperparameters.ending
                            + ".npy"
                        )
                    )
                    * b2.volt
                )
            else:
                self.neuron_groups["e"].theta = (
                    np.ones((self.experiment_hyperparameters.n_e,)) * 20.0 * b2.mV
                )

            print("create recurrent connections")

            conn_type: str

            for (
                conn_type
            ) in self.network_architecture_hyperparameters.recurrent_conn_names:
                conn_name = name + conn_type[0] + name + conn_type[1]
                weight_matrix = get_matrix_from_file(
                    self.experiment_hyperparameters.weight_path
                    / ".."
                    / "random"
                    / (conn_name + self.experiment_hyperparameters.ending + ".npy"),
                    self.experiment_hyperparameters.ending,
                    self.experiment_hyperparameters.n_input,
                    self.experiment_hyperparameters.n_e,
                    self.experiment_hyperparameters.n_i,
                )
                model = "w : 1"
                pre = f"g{conn_type[0]}_post += w"
                post = ""

                if self.experiment_hyperparameters.ee_stdp_on:
                    if (
                        "ee"
                        in self.network_architecture_hyperparameters.recurrent_conn_names
                    ):
                        model += self.model_equations.eqs_stdp_ee
                        pre += "; " + self.model_equations.eqs_stdp_pre_ee
                        post = self.model_equations.eqs_stdp_post_ee

                self.connections[conn_name] = b2.Synapses(
                    self.neuron_groups[conn_name[0:2]],
                    self.neuron_groups[conn_name[2:4]],
                    model=model,
                    on_pre=pre,
                    on_post=post,
                )
                self.connections[conn_name].connect(True)  # all-to-all connection
                self.connections[conn_name].w = weight_matrix[
                    self.connections[conn_name].i, self.connections[conn_name].j
                ]

            print("create monitors for", name)
            self.rate_monitors[name + "e"] = b2.PopulationRateMonitor(
                self.neuron_groups[name + "e"]
            )
            self.rate_monitors[name + "i"] = b2.PopulationRateMonitor(
                self.neuron_groups[name + "i"]
            )
            self.spike_counters[name + "e"] = b2.SpikeMonitor(
                self.neuron_groups[name + "e"], record=False
            )

            if self.experiment_hyperparameters.record_spikes:
                self.spike_monitors[name + "e"] = b2.SpikeMonitor(
                    self.neuron_groups[name + "e"]
                )
                self.spike_monitors[name + "i"] = b2.SpikeMonitor(
                    self.neuron_groups[name + "i"]
                )

    def create_input_population_and_connection(self):
        # ------------------------------------------------------------------------------
        # create input population and connections from input populations
        # ------------------------------------------------------------------------------
        i: int
        name: str

        for i, name in enumerate(
            self.network_architecture_hyperparameters.input_population_names
        ):
            self.input_groups[name + "e"] = b2.PoissonGroup(
                self.experiment_hyperparameters.n_input, 0 * b2.Hz
            )
            self.rate_monitors[name + "e"] = b2.PopulationRateMonitor(
                self.input_groups[name + "e"]
            )

        for name in self.network_architecture_hyperparameters.input_connection_names:
            print("create connections between", name[0], "and", name[1])

            for conn_type in self.network_architecture_hyperparameters.input_conn_names:
                conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
                weight_matrix = get_matrix_from_file(
                    self.experiment_hyperparameters.weight_path
                    / (conn_name + self.experiment_hyperparameters.ending + ".npy"),
                    self.experiment_hyperparameters.ending,
                    self.experiment_hyperparameters.n_input,
                    self.experiment_hyperparameters.n_e,
                    self.experiment_hyperparameters.n_i,
                )
                model = "w : 1"
                pre = f"g{conn_type[0]}_post += w"
                post = ""

                if self.experiment_hyperparameters.ee_stdp_on:
                    print("create STDP for connection", name[0] + "e" + name[1] + "e")
                    model += self.model_equations.eqs_stdp_ee
                    pre += "; " + self.model_equations.eqs_stdp_pre_ee
                    post = self.model_equations.eqs_stdp_post_ee

                self.connections[conn_name] = b2.Synapses(
                    self.input_groups[conn_name[0:2]],
                    self.neuron_groups[conn_name[2:4]],
                    model=model,
                    on_pre=pre,
                    on_post=post,
                )
                min_delay = self.network_architecture_hyperparameters.delay[conn_type][
                    0
                ]
                max_delay = self.network_architecture_hyperparameters.delay[conn_type][
                    1
                ]
                delta_delay = max_delay - min_delay

                self.connections[conn_name].connect(True)  # all-to-all connection
                self.connections[conn_name].delay = "min_delay + rand() * delta_delay"
                self.connections[conn_name].w = weight_matrix[
                    self.connections[conn_name].i, self.connections[conn_name].j
                ]

    def run(self):
        # ------------------------------------------------------------------------------
        # run the simulation and set inputs
        # ------------------------------------------------------------------------------

        net = b2.Network()

        for obj_list in [
            self.neuron_groups,
            self.input_groups,
            self.connections,
            self.rate_monitors,
            self.spike_monitors,
            self.spike_counters,
        ]:
            for key in obj_list:
                net.add(obj_list[key])

        previous_spike_count: np.ndarray = np.zeros(self.experiment_hyperparameters.n_e)
        assignments: np.ndarray = np.zeros(self.experiment_hyperparameters.n_e)
        input_numbers: List[int] = [0] * self.experiment_hyperparameters.num_examples
        output_numbers: np.ndarray = np.zeros(
            (self.experiment_hyperparameters.num_examples, 10)
        )

        input_weight_monitor: AxesImage | None = None
        fig_weights: Figure | None = None

        if not self.experiment_hyperparameters.test_mode:
            input_weight_monitor, fig_weights = plot_2d_input_weights(
                self.experiment_hyperparameters.n_input,
                self.experiment_hyperparameters.n_e,
                self.connections,
                self.fig_num,
                self.synapse_model_hyperparameters.wmax_ee,
            )
            self.fig_num += 1

        # TODO: extract
        performance_monitor: Line2D | None = None
        performance: np.ndarray | None = None
        fig_performance: b2.Figure | None = None

        if self.experiment_hyperparameters.do_plot_performance:
            (
                performance_monitor,
                performance,
                self.fig_num,
                fig_performance,
            ) = plot_performance(
                self.fig_num,
                self.experiment_hyperparameters.num_examples,
                self.experiment_hyperparameters.update_interval,
            )

        for i, name in enumerate(
            self.network_architecture_hyperparameters.input_population_names
        ):
            self.input_groups[name + "e"].rates = 0 * b2.Hz

        equation_variables = {
            "v_rest_e": self.neuron_model_hyperparameters.v_rest_e,
            "v_rest_i": self.neuron_model_hyperparameters.v_rest_i,
            "v_thresh_e": self.neuron_model_hyperparameters.v_thresh_e,
            "v_thresh_i": self.neuron_model_hyperparameters.v_thresh_i,
            "refrac_e": self.neuron_model_hyperparameters.refrac_e,
            "v_reset_e": self.neuron_model_hyperparameters.v_rest_e,
            "v_reset_i": self.neuron_model_hyperparameters.v_reset_i,
            "nu_ee_pre": self.synapse_model_hyperparameters.nu_ee_pre,
            "tc_post_1_ee": self.synapse_model_hyperparameters.tc_post_1_ee,
            "tc_post_2_ee": self.synapse_model_hyperparameters.tc_post_2_ee,
            "tc_pre_ee": self.synapse_model_hyperparameters.tc_pre_ee,
            "wmax_ee": self.synapse_model_hyperparameters.wmax_ee,
            "nu_ee_post": self.synapse_model_hyperparameters.nu_ee_post,
            "offset": self.synapse_model_hyperparameters.offset,
        }

        if not self.experiment_hyperparameters.test_mode:
            equation_variables.update(
                {
                    "tc_theta": self.synapse_model_hyperparameters.tc_theta,
                    "theta_plus_e": self.synapse_model_hyperparameters.theta_plus_e,
                }
            )

        net.run(0 * b2.second, namespace=equation_variables)
        iteration: int = 0

        while iteration < (int(self.experiment_hyperparameters.num_examples)):
            if self.experiment_hyperparameters.test_mode:
                if self.experiment_hyperparameters.use_testing_set:
                    spike_rates = (
                        self.testing_data["x"][iteration % 10000, :, :].reshape(
                            (self.experiment_hyperparameters.n_input,)
                        )
                        / 8.0
                        * self.network_architecture_hyperparameters.input_intensity
                    )
                else:
                    spike_rates = (
                        self.training_data["x"][iteration % 60000, :, :].reshape(
                            (self.experiment_hyperparameters.n_input,)
                        )
                        / 8.0
                        * self.network_architecture_hyperparameters.input_intensity
                    )
            else:
                self.normalize_weights()
                spike_rates = (
                    self.training_data["x"][iteration % 60000, :, :].reshape(
                        (self.experiment_hyperparameters.n_input,)
                    )
                    / 8.0
                    * self.network_architecture_hyperparameters.input_intensity
                )

            self.input_groups["Xe"].rates = spike_rates * b2.Hz
            #     print 'run number:', j+1, 'of', int(num_examples)
            net.run(
                self.experiment_hyperparameters.single_example_time,
                report="text",
                namespace=equation_variables,
            )

            if (
                iteration % self.experiment_hyperparameters.update_interval == 0
                and iteration > 0
            ):
                assignments = self.get_new_assignments(
                    self.result_monitor[:],
                    input_numbers[
                        iteration
                        - self.experiment_hyperparameters.update_interval : iteration
                    ],
                )

            # TODO: extract
            if (
                iteration % self.experiment_hyperparameters.weight_update_interval == 0
                and not self.experiment_hyperparameters.test_mode
            ):
                update_2d_input_weights(
                    input_weight_monitor,
                    fig_weights,
                    self.experiment_hyperparameters.n_input,
                    self.experiment_hyperparameters.n_e,
                    self.connections,
                )
                b2.pause(0.1)  # triggers update of the plots

            if (
                iteration % self.experiment_hyperparameters.save_connections_interval
                == 0
                and iteration > 0
                and not self.experiment_hyperparameters.test_mode
            ):
                save_connections(
                    self.network_architecture_hyperparameters.save_conns,
                    self.connections,
                    self.experiment_hyperparameters.data_path,
                    str(iteration),
                )
                save_theta(
                    self.network_architecture_hyperparameters.population_names,
                    self.experiment_hyperparameters.data_path,
                    self.neuron_groups,
                    str(iteration),
                )

            current_spike_count: np.ndarray = (
                np.asarray(self.spike_counters["Ae"].count[:]) - previous_spike_count
            )
            previous_spike_count: np.ndarray = np.copy(
                self.spike_counters["Ae"].count[:]
            )

            if np.sum(current_spike_count) < 5:
                self.network_architecture_hyperparameters.input_intensity += 1

                for i, name in enumerate(
                    self.network_architecture_hyperparameters.input_population_names
                ):
                    self.input_groups[name + "e"].rates = 0 * b2.Hz

                net.run(
                    self.experiment_hyperparameters.resting_time,
                    namespace=equation_variables,
                )
            else:
                self.result_monitor[
                    iteration % self.experiment_hyperparameters.update_interval, :
                ] = current_spike_count

                if (
                    self.experiment_hyperparameters.test_mode
                    and self.experiment_hyperparameters.use_testing_set
                ):
                    input_numbers[iteration] = self.testing_data["y"][
                        iteration % 10000
                    ][0].item()
                else:
                    input_numbers[iteration] = self.training_data["y"][
                        iteration % 60000
                    ][0].item()

                output_numbers[iteration, :] = self.get_recognized_number_ranking(
                    assignments,
                    self.result_monitor[
                        iteration % self.experiment_hyperparameters.update_interval, :
                    ],
                )

                if iteration % 100 == 0 and iteration > 0:
                    print(
                        "runs done:",
                        iteration,
                        "of",
                        int(self.experiment_hyperparameters.num_examples),
                    )

                # TODO: extract
                if (
                    iteration % self.experiment_hyperparameters.update_interval == 0
                    and iteration > 0
                ):
                    if self.experiment_hyperparameters.do_plot_performance:
                        unused, performance = update_performance_plot(
                            performance_monitor,
                            performance,
                            iteration,
                            fig_performance,
                            self.experiment_hyperparameters.update_interval,
                            output_numbers,
                            input_numbers,
                        )
                        print(
                            "Classification performance",
                            performance[
                                : (
                                    int(
                                        iteration
                                        / float(
                                            self.experiment_hyperparameters.update_interval
                                        )
                                    )
                                )
                                + 1
                            ],
                        )

                for i, name in enumerate(
                    self.network_architecture_hyperparameters.input_population_names
                ):
                    self.input_groups[name + "e"].rates = 0 * b2.Hz

                net.run(
                    self.experiment_hyperparameters.resting_time,
                    namespace=equation_variables,
                )
                self.network_architecture_hyperparameters.input_intensity = (
                    self.network_architecture_hyperparameters.start_input_intensity
                )
                iteration += 1

        # ------------------------------------------------------------------------------
        # save results
        # ------------------------------------------------------------------------------
        print("save results")
        if not self.experiment_hyperparameters.test_mode:
            save_theta(
                self.network_architecture_hyperparameters.population_names,
                self.experiment_hyperparameters.data_path,
                self.neuron_groups,
            )
        if not self.experiment_hyperparameters.test_mode:
            save_connections(
                self.network_architecture_hyperparameters.save_conns,
                self.connections,
                self.experiment_hyperparameters.data_path,
            )
        else:
            np.save(
                self.experiment_hyperparameters.data_path
                / "activity"
                / ("resultPopVecs" + str(self.experiment_hyperparameters.num_examples)),
                self.result_monitor,
            )
            np.save(
                self.experiment_hyperparameters.data_path
                / "activity"
                / ("inputNumbers" + str(self.experiment_hyperparameters.num_examples)),
                input_numbers,
            )

        plot_results(
            fig_num=self.fig_num,
            rate_monitors=self.rate_monitors,
            spike_monitors=self.spike_monitors,
            spike_counters=self.spike_counters,
            connections=self.connections,
            n_input=self.experiment_hyperparameters.n_input,
            n_e=self.experiment_hyperparameters.n_e,
            weight_max_ee=self.synapse_model_hyperparameters.wmax_ee,
        )


def main():
    Runner().run()


if __name__ == "__main__":
    main()

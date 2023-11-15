"""
Created on 15.12.2014

@author: Peter U. Diehl
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import brian2 as b2
import numpy as np

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
    ee_STDP_on: bool
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
            num_examples = 100 * 1
            use_testing_set = True
            do_plot_performance = False
            record_spikes = True
            ee_STDP_on = False
            update_interval = num_examples
        else:
            weight_path = data_path / "random"
            num_examples = 600 * 3
            use_testing_set = False
            do_plot_performance = True

            if num_examples <= 60000:
                record_spikes = True
            else:
                record_spikes = True

            ee_STDP_on = True

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
            ee_STDP_on=ee_STDP_on,
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
            test_mode=True
        )

        self.neuron_model_hyperparameters = NeuronModelHyperparameters.get_default()

        self.weight = {}
        self.delay = {}
        self.input_population_names = ["X"]
        self.population_names = ["A"]
        self.input_connection_names = ["XA"]
        self.save_conns = ["XeAe"]
        self.input_conn_names = ["ee_input"]
        self.recurrent_conn_names = ["ei", "ie"]
        self.weight["ee_input"] = 78.0
        self.delay["ee_input"] = (0 * b2.ms, 10 * b2.ms)
        self.delay["ei_input"] = (0 * b2.ms, 5 * b2.ms)
        self.input_intensity = 2.0
        self.start_input_intensity = self.input_intensity

        self.synapse_model_hyperparameters = SynapseModelHyperparameters.get_default()

        if self.experiment_hyperparameters.test_mode:
            self.scr_e = "v = v_reset_e; timer = 0*ms"
        else:
            self.scr_e = "v = v_reset_e; theta += theta_plus_e; timer = 0*ms"

        self.v_thresh_e_eqs = "(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)"
        self.v_thresh_i_eqs = "v>v_thresh_i"
        self.v_reset_i_eqs = "v=v_reset_i"

        self.neuron_eqs_e = """
                       dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
                       I_synE = ge * nS *         -v                           : amp
                       I_synI = gi * nS * (-100.*mV-v)                          : amp
                       dge/dt = -ge/(1.0*ms)                                   : 1
                       dgi/dt = -gi/(2.0*ms)                                  : 1
                       """

        if self.experiment_hyperparameters.test_mode:
            self.neuron_eqs_e += "\n  theta      :volt"
        else:
            self.neuron_eqs_e += "\n  dtheta/dt = -theta / (tc_theta)  : volt"
        self.neuron_eqs_e += "\n  dtimer/dt = 0.1  : second"

        self.neuron_eqs_i = """
                dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
                I_synE = ge * nS *         -v                           : amp
                I_synI = gi * nS * (-85.*mV-v)                          : amp
                dge/dt = -ge/(1.0*ms)                                   : 1
                dgi/dt = -gi/(2.0*ms)                                  : 1
                """
        self.eqs_stdp_ee = """
                        post2before                            : 1
                        dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                        dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                        dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
                    """
        self.eqs_stdp_pre_ee = "pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)"
        self.eqs_stdp_post_ee = "post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1."

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
            self.experiment_hyperparameters.n_e * len(self.population_names),
            self.neuron_eqs_e,
            threshold=self.v_thresh_e_eqs,
            refractory=self.neuron_model_hyperparameters.refrac_e,
            reset=self.scr_e,
            method="euler",
        )
        self.neuron_groups["i"] = b2.NeuronGroup(
            self.experiment_hyperparameters.n_i * len(self.population_names),
            self.neuron_eqs_i,
            threshold=self.v_thresh_i_eqs,
            refractory=self.neuron_model_hyperparameters.refrac_i,
            reset=self.v_reset_i_eqs,
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
                colSums = np.sum(temp_conn, axis=0)
                colFactors = self.weight["ee_input"] / colSums

                for j in range(self.experiment_hyperparameters.n_e):  #
                    temp_conn[:, j] *= colFactors[j]

                self.connections[connName].w = temp_conn[
                    self.connections[connName].i, self.connections[connName].j
                ]

    def get_recognized_number_ranking(self, assignments, spike_rates):
        summed_rates = [0] * 10
        num_assignments = [0] * 10
        for i in range(10):
            num_assignments[i] = len(np.where(assignments == i)[0])
            if num_assignments[i] > 0:
                summed_rates[i] = (
                    np.sum(spike_rates[assignments == i]) / num_assignments[i]
                )
        return np.argsort(summed_rates)[::-1]

    def get_new_assignments(self, result_monitor, input_numbers):
        assignments = np.zeros(self.experiment_hyperparameters.n_e)
        input_nums = np.asarray(input_numbers)
        maximum_rate = [0] * self.experiment_hyperparameters.n_e

        for j in range(10):
            num_assignments = len(np.where(input_nums == j)[0])

            if num_assignments > 0:
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
        for subgroup_n, name in enumerate(self.population_names):
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
                    np.ones((self.experiment_hyperparameters.n_e)) * 20.0 * b2.mV
                )

            print("create recurrent connections")

            for conn_type in self.recurrent_conn_names:
                connName = name + conn_type[0] + name + conn_type[1]
                weightMatrix = get_matrix_from_file(
                    self.experiment_hyperparameters.weight_path
                    / ".."
                    / "random"
                    / (connName + self.experiment_hyperparameters.ending + ".npy"),
                    self.experiment_hyperparameters.ending,
                    self.experiment_hyperparameters.n_input,
                    self.experiment_hyperparameters.n_e,
                    self.experiment_hyperparameters.n_i,
                )
                model = "w : 1"
                pre = "g%s_post += w" % conn_type[0]
                post = ""

                if self.experiment_hyperparameters.ee_STDP_on:
                    if "ee" in self.recurrent_conn_names:
                        model += self.eqs_stdp_ee
                        pre += "; " + self.eqs_stdp_pre_ee
                        post = self.eqs_stdp_post_ee

                self.connections[connName] = b2.Synapses(
                    self.neuron_groups[connName[0:2]],
                    self.neuron_groups[connName[2:4]],
                    model=model,
                    on_pre=pre,
                    on_post=post,
                )
                self.connections[connName].connect(True)  # all-to-all connection
                self.connections[connName].w = weightMatrix[
                    self.connections[connName].i, self.connections[connName].j
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
        pop_values = [0, 0, 0]

        for i, name in enumerate(self.input_population_names):
            self.input_groups[name + "e"] = b2.PoissonGroup(
                self.experiment_hyperparameters.n_input, 0 * b2.Hz
            )
            self.rate_monitors[name + "e"] = b2.PopulationRateMonitor(
                self.input_groups[name + "e"]
            )

        for name in self.input_connection_names:
            print("create connections between", name[0], "and", name[1])

            for connType in self.input_conn_names:
                connName = name[0] + connType[0] + name[1] + connType[1]
                weightMatrix = get_matrix_from_file(
                    self.experiment_hyperparameters.weight_path
                    / (connName + self.experiment_hyperparameters.ending + ".npy"),
                    self.experiment_hyperparameters.ending,
                    self.experiment_hyperparameters.n_input,
                    self.experiment_hyperparameters.n_e,
                    self.experiment_hyperparameters.n_i,
                )
                model = "w : 1"
                pre = "g%s_post += w" % connType[0]
                post = ""

                if self.experiment_hyperparameters.ee_STDP_on:
                    print("create STDP for connection", name[0] + "e" + name[1] + "e")
                    model += self.eqs_stdp_ee
                    pre += "; " + self.eqs_stdp_pre_ee
                    post = self.eqs_stdp_post_ee

                self.connections[connName] = b2.Synapses(
                    self.input_groups[connName[0:2]],
                    self.neuron_groups[connName[2:4]],
                    model=model,
                    on_pre=pre,
                    on_post=post,
                )
                minDelay = self.delay[connType][0]
                maxDelay = self.delay[connType][1]
                deltaDelay = maxDelay - minDelay

                self.connections[connName].connect(True)  # all-to-all connection
                self.connections[connName].delay = "minDelay + rand() * deltaDelay"
                self.connections[connName].w = weightMatrix[
                    self.connections[connName].i, self.connections[connName].j
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

        previous_spike_count = np.zeros(self.experiment_hyperparameters.n_e)
        assignments = np.zeros(self.experiment_hyperparameters.n_e)
        input_numbers = [0] * self.experiment_hyperparameters.num_examples
        outputNumbers = np.zeros((self.experiment_hyperparameters.num_examples, 10))

        if not self.experiment_hyperparameters.test_mode:
            input_weight_monitor, fig_weights = plot_2d_input_weights(
                self.experiment_hyperparameters.n_input,
                self.experiment_hyperparameters.n_e,
                self.connections,
                self.fig_num,
                self.synapse_model_hyperparameters.wmax_ee,
            )
            self.fig_num += 1

        if self.experiment_hyperparameters.do_plot_performance:
            # TODO: make these into instance variables?
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

        for i, name in enumerate(self.input_population_names):
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
        j: int = 0

        while j < (int(self.experiment_hyperparameters.num_examples)):
            if self.experiment_hyperparameters.test_mode:
                if self.experiment_hyperparameters.use_testing_set:
                    spike_rates = (
                        self.testing_data["x"][j % 10000, :, :].reshape(
                            (self.experiment_hyperparameters.n_input,)
                        )
                        / 8.0
                        * self.input_intensity
                    )
                else:
                    spike_rates = (
                        self.training_data["x"][j % 60000, :, :].reshape(
                            (self.experiment_hyperparameters.n_input,)
                        )
                        / 8.0
                        * self.input_intensity
                    )
            else:
                self.normalize_weights()
                spike_rates = (
                    self.training_data["x"][j % 60000, :, :].reshape(
                        (self.experiment_hyperparameters.n_input,)
                    )
                    / 8.0
                    * self.input_intensity
                )

            self.input_groups["Xe"].rates = spike_rates * b2.Hz
            #     print 'run number:', j+1, 'of', int(num_examples)
            net.run(
                self.experiment_hyperparameters.single_example_time,
                report="text",
                namespace=equation_variables,
            )

            if j % self.experiment_hyperparameters.update_interval == 0 and j > 0:
                assignments = self.get_new_assignments(
                    self.result_monitor[:],
                    input_numbers[
                        j - self.experiment_hyperparameters.update_interval : j
                    ],
                )

            if (
                j % self.experiment_hyperparameters.weight_update_interval == 0
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
                j % self.experiment_hyperparameters.save_connections_interval == 0
                and j > 0
                and not self.experiment_hyperparameters.test_mode
            ):
                save_connections(
                    self.save_conns,
                    self.connections,
                    self.experiment_hyperparameters.data_path,
                    str(j),
                )
                save_theta(
                    self.population_names,
                    self.experiment_hyperparameters.data_path,
                    self.neuron_groups,
                    str(j),
                )

            current_spike_count = (
                np.asarray(self.spike_counters["Ae"].count[:]) - previous_spike_count
            )
            previous_spike_count = np.copy(self.spike_counters["Ae"].count[:])

            if np.sum(current_spike_count) < 5:
                self.input_intensity += 1

                for i, name in enumerate(self.input_population_names):
                    self.input_groups[name + "e"].rates = 0 * b2.Hz

                net.run(
                    self.experiment_hyperparameters.resting_time,
                    namespace=equation_variables,
                )
            else:
                self.result_monitor[
                    j % self.experiment_hyperparameters.update_interval, :
                ] = current_spike_count

                if (
                    self.experiment_hyperparameters.test_mode
                    and self.experiment_hyperparameters.use_testing_set
                ):
                    input_numbers[j] = self.testing_data["y"][j % 10000][0]
                else:
                    input_numbers[j] = self.training_data["y"][j % 60000][0]

                outputNumbers[j, :] = self.get_recognized_number_ranking(
                    assignments,
                    self.result_monitor[
                        j % self.experiment_hyperparameters.update_interval, :
                    ],
                )

                if j % 100 == 0 and j > 0:
                    print(
                        "runs done:",
                        j,
                        "of",
                        int(self.experiment_hyperparameters.num_examples),
                    )

                if j % self.experiment_hyperparameters.update_interval == 0 and j > 0:
                    if self.experiment_hyperparameters.do_plot_performance:
                        unused, performance = update_performance_plot(
                            performance_monitor,
                            performance,
                            j,
                            fig_performance,
                            self.experiment_hyperparameters.update_interval,
                            outputNumbers,
                            input_numbers,
                        )
                        print(
                            "Classification performance",
                            performance[
                                : (
                                    int(
                                        j
                                        / float(
                                            self.experiment_hyperparameters.update_interval
                                        )
                                    )
                                )
                                + 1
                            ],
                        )

                for i, name in enumerate(self.input_population_names):
                    self.input_groups[name + "e"].rates = 0 * b2.Hz

                net.run(
                    self.experiment_hyperparameters.resting_time,
                    namespace=equation_variables,
                )
                self.input_intensity = self.start_input_intensity
                j += 1

        # ------------------------------------------------------------------------------
        # save results
        # ------------------------------------------------------------------------------
        print("save results")
        if not self.experiment_hyperparameters.test_mode:
            save_theta(
                self.population_names,
                self.experiment_hyperparameters.data_path,
                self.neuron_groups,
            )
        if not self.experiment_hyperparameters.test_mode:
            save_connections(
                self.save_conns,
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

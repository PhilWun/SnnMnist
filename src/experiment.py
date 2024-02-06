"""
Created on 15.12.2014

@author: Peter U. Diehl
"""
import time
from typing import Dict, List

import brian2 as b2
import numpy as np

from src.data_handler import (
    get_labeled_data,
    get_matrix_from_file,
    save_connections,
    save_theta,
)
from src.hyperparameters import (
    ExperimentHyperparameters,
    NetworkArchitectureHyperparameters,
    NeuronModelHyperparameters,
    SynapseModelHyperparameters,
    ModelEquations,
)
from src.plotting import PlottingHandler


# noinspection PyInterpreter
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
        self.testing_data = get_labeled_data("testing", training=False)
        end = time.time()
        print("time needed to load test set:", end - start)

        # ------------------------------------------------------------------------------
        # set parameters and equations
        # ------------------------------------------------------------------------------
        self.exp_hyper = ExperimentHyperparameters.get_default(test_mode=False)
        self.exp_hyper.num_examples = 10000
        self.exp_hyper.update_interval = 100
        self.exp_hyper.weight_update_interval = 100
        # self.exp_hyper.file_postfix = "_refrac_factor"

        self.neuron_hyper = NeuronModelHyperparameters.get_default()
        self.net_hyper = NetworkArchitectureHyperparameters.get_default()
        # self.net_hyper.delay = {
        #     "ee_input": (0 * b2.ms, 0 * b2.ms),
        #     "ei_input": (0 * b2.ms, 0 * b2.ms),
        # }

        self.syn_hyper = SynapseModelHyperparameters.get_default()
        self.model_equations = ModelEquations.get_default(self.exp_hyper.test_mode)

        self.neuron_groups = {}
        self.input_groups: Dict[str, b2.PoissonGroup] = {}
        self.connections: Dict[str, b2.Synapses] = {}
        self.rate_monitors: Dict[str, b2.PopulationRateMonitor] = {}
        self.spike_monitors: Dict[str, b2.SpikeMonitor] = {}
        self.spike_counters: Dict[str, b2.SpikeMonitor] = {}
        self.result_monitor: np.ndarray = np.zeros(
            (
                self.exp_hyper.update_interval,
                self.exp_hyper.n_e,
            )
        )
        """stores the latest spike counts from subgroup Ae"""

        self.neuron_groups["e"] = b2.NeuronGroup(
            self.exp_hyper.n_e * len(self.net_hyper.population_names),
            self.model_equations.neuron_eqs_e,
            threshold=self.model_equations.v_thresh_e_eqs,
            refractory=self.neuron_hyper.refrac_e,
            reset=self.model_equations.scr_e,
            method="euler",
        )
        self.neuron_groups["i"] = b2.NeuronGroup(
            self.exp_hyper.n_i * len(self.net_hyper.population_names),
            self.model_equations.neuron_eqs_i,
            threshold=self.model_equations.v_thresh_i_eqs,
            refractory=self.neuron_hyper.refrac_i,
            reset=self.model_equations.v_reset_i_eqs,
            method="euler",
        )

        self.create_network_and_recurrent_connections()
        self.create_input_population_and_connection()

        self.plotting_handler = PlottingHandler()

    def normalize_weights(self):
        """
        Normalize the weights from input to Ae so that the weights to each neuron in Ae have a specific sum.
        """
        for connName in self.connections:
            if not (connName[1] == "e" and connName[3] == "e"):
                continue

            len_source = len(self.connections[connName].source)
            len_target = len(self.connections[connName].target)
            connection = np.zeros((len_source, len_target))
            connection[
                self.connections[connName].i, self.connections[connName].j
            ] = self.connections[connName].w
            temp_conn = np.copy(connection)
            col_sums = np.sum(temp_conn, axis=0)
            col_factors = self.net_hyper.weight["ee_input"] / col_sums

            for j in range(self.exp_hyper.n_e):  #
                temp_conn[:, j] *= col_factors[j]

            self.connections[connName].w = temp_conn[
                self.connections[connName].i, self.connections[connName].j
            ]

    @staticmethod
    def get_recognized_number_ranking(
        assignments: np.ndarray, spike_rates: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the prediction which classes are most-likely.

        :param assignments: class assignments of the excitatory neurons, shape: (neuron count)
        :param spike_rates: spike counts of the excitatory neurons for the current input, shape: (neuron count)
        :return: ranking of the most likely classes, most-likely first, shape: (10)
        """
        summed_rates = [0] * 10
        num_assignments = [0] * 10

        for i in range(10):
            num_assignments[i] = len(np.where(assignments == i)[0])

            if num_assignments[i] > 0:
                summed_rates[i] = (
                    np.sum(spike_rates[assignments == i]) / num_assignments[i]
                )

        return np.argsort(summed_rates)[::-1]

    def get_new_assignments(
        self, result_monitor: np.ndarray, input_numbers: List[int]
    ) -> np.ndarray:
        """
        Assigns every neuron the class for which it spiked the most.

        :param result_monitor: spike count for each neuron, shape: (update interval, neuron count)
        :param input_numbers: target classes, shape: (update interval)
        :return: class assignments for each neuron
        """
        assignments: np.ndarray = np.zeros(self.exp_hyper.n_e)
        input_nums: np.ndarray = np.asarray(input_numbers)
        maximum_rate: List[float] = [0] * self.exp_hyper.n_e

        for j in range(10):
            num_assignments = len(np.where(input_nums == j)[0])
            assert num_assignments > 0

            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments

            for i in range(self.exp_hyper.n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j

        return assignments

    def create_network_and_recurrent_connections(self) -> None:
        """
        Create the neuron subgroups, set the start state, create the connections between excitatory and inhibitory
        subgroups, add spike and rate monitors.

        :return: None
        """

        for subgroup_n, name in enumerate(self.net_hyper.population_names):
            print("create neuron group", name)

            self.neuron_groups[name + "e"] = self.neuron_groups["e"][
                subgroup_n * self.exp_hyper.n_e : (subgroup_n + 1) * self.exp_hyper.n_e
            ]
            self.neuron_groups[name + "i"] = self.neuron_groups["i"][
                subgroup_n * self.exp_hyper.n_i : (subgroup_n + 1) * self.exp_hyper.n_e
            ]

            self.neuron_groups[name + "e"].v = (
                self.neuron_hyper.v_rest_e + self.neuron_hyper.v_start_offset
            )
            self.neuron_groups[name + "i"].v = (
                self.neuron_hyper.v_rest_i + self.neuron_hyper.v_start_offset
            )

            self.neuron_groups["e"].theta = self.generate_or_load_theta(name)

            print("create recurrent connections")

            for conn_type in self.net_hyper.recurrent_conn_names:
                conn_name = name + conn_type[0] + name + conn_type[1]
                weight_matrix = self.generate_or_load_weights(conn_name)

                model = self.model_equations.syn_eqs
                on_post = self.model_equations.syn_eqs_post

                if conn_type[0] == "e":
                    on_pre = self.model_equations.syn_eqs_pre_e
                elif conn_type[0] == "i":
                    on_pre = self.model_equations.syn_eqs_pre_i
                else:
                    raise ValueError

                if self.exp_hyper.ee_stdp_on:
                    # not true: STDP is not used for the recurrent connections
                    if "ee" in self.net_hyper.recurrent_conn_names:
                        model += self.model_equations.eqs_stdp_ee
                        on_pre += "; " + self.model_equations.eqs_stdp_pre_ee
                        on_post = self.model_equations.eqs_stdp_post_ee

                self.connections[conn_name] = b2.Synapses(
                    self.neuron_groups[conn_name[0:2]],
                    self.neuron_groups[conn_name[2:4]],
                    model=model,
                    on_pre=on_pre,
                    on_post=on_post,
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

            if self.exp_hyper.record_spikes:
                self.spike_monitors[name + "e"] = b2.SpikeMonitor(
                    self.neuron_groups[name + "e"]
                )
                self.spike_monitors[name + "i"] = b2.SpikeMonitor(
                    self.neuron_groups[name + "i"]
                )

    def create_input_population_and_connection(self) -> None:
        """
        Create input groups, add rate monitors, add connections to neuron groups

        :return: None
        """

        for i, name in enumerate(self.net_hyper.input_population_names):
            self.input_groups[name + "e"] = b2.PoissonGroup(
                self.exp_hyper.n_input, 0 * b2.Hz
            )
            self.rate_monitors[name + "e"] = b2.PopulationRateMonitor(
                self.input_groups[name + "e"]
            )

        for name in self.net_hyper.input_connection_names:
            print("create connections between", name[0], "and", name[1])

            for conn_type in self.net_hyper.input_conn_names:
                conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
                weight_matrix = self.generate_or_load_weights(conn_name)

                model = self.model_equations.syn_eqs
                on_post = self.model_equations.syn_eqs_post

                if conn_type[0] == "e":
                    on_pre = self.model_equations.syn_eqs_pre_e
                elif conn_type[0] == "i":
                    on_pre = self.model_equations.syn_eqs_pre_i
                else:
                    raise ValueError

                if self.exp_hyper.ee_stdp_on:
                    print("create STDP for connection", name[0] + "e" + name[1] + "e")
                    model += self.model_equations.eqs_stdp_ee
                    on_pre += "; " + self.model_equations.eqs_stdp_pre_ee
                    on_post = self.model_equations.eqs_stdp_post_ee

                self.connections[conn_name] = b2.Synapses(
                    self.input_groups[conn_name[0:2]],
                    self.neuron_groups[conn_name[2:4]],
                    model=model,
                    on_pre=on_pre,
                    on_post=on_post,
                )
                min_delay = self.net_hyper.delay[conn_type][0]
                max_delay = self.net_hyper.delay[conn_type][1]
                delta_delay = max_delay - min_delay

                self.connections[conn_name].connect(True)  # all-to-all connection
                self.connections[conn_name].delay = "min_delay + rand() * delta_delay"
                self.connections[conn_name].w = weight_matrix[
                    self.connections[conn_name].i, self.connections[conn_name].j
                ]

    def generate_or_load_weights(self, conn_name: str) -> np.ndarray:
        """
        Generate initial weights in training mode. Load saved weights in test mode and generate the rest.

        :param conn_name: name of the connection for which the weights should be generated or loaded
        :return: generated of loaded weights, shape: (presynaptic neuron count, postsynaptic neuron count)
        """

        load_from_file = (
            conn_name in self.net_hyper.save_conns and self.exp_hyper.test_mode
        )

        if load_from_file:
            return get_matrix_from_file(
                self.exp_hyper.weight_path
                / (conn_name + self.exp_hyper.file_postfix + ".npy"),
                self.exp_hyper.file_postfix,
                self.exp_hyper.n_input,
                self.exp_hyper.n_e,
                self.exp_hyper.n_i,
            )
        else:
            if conn_name == "AeAi":
                assert self.exp_hyper.n_e == self.exp_hyper.n_i
                return np.eye(self.exp_hyper.n_e) * 10.4
            elif conn_name == "AiAe":
                assert self.exp_hyper.n_e == self.exp_hyper.n_i
                n = self.exp_hyper.n_e
                return (np.ones((n, n)) - np.eye(n)) * 17.0
            elif conn_name == "XeAe":
                return (
                    np.random.random((self.exp_hyper.n_input, self.exp_hyper.n_e)) * 0.3
                )
            else:
                raise ValueError(f"unknown connection name {conn_name}")

    def generate_or_load_theta(self, sub_group_name: str) -> np.ndarray:
        """
        Generate initial values for the dynamic threshold offset theta in training mode. Load them from file in test
        mode.

        :param sub_group_name: name of the subgroup of neurons for which the theta values should be generated or loaded
        :return: theta values, shape: (neuron count)
        """

        load_from_file = self.exp_hyper.test_mode

        if load_from_file:
            return (
                np.load(
                    self.exp_hyper.weight_path
                    / f"theta_{sub_group_name}{self.exp_hyper.file_postfix}.npy"
                )
                * b2.volt
            )
        else:
            return np.ones((self.exp_hyper.n_e,)) * self.syn_hyper.theta_start

    def run(self) -> None:
        """
        Run the experiment, train / test the model.

        :return: None
        """

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

        previous_spike_count: np.ndarray = np.zeros(self.exp_hyper.n_e)
        assignments: np.ndarray = np.zeros(self.exp_hyper.n_e)
        input_numbers: List[int] = [0] * self.exp_hyper.num_examples
        """the latest target values"""
        output_numbers: np.ndarray = np.zeros((self.exp_hyper.num_examples, 10))
        """the latest output values"""

        self.plotting_handler.plot_input_weights(
            self.exp_hyper.test_mode,
            self.exp_hyper.n_input,
            self.exp_hyper.n_e,
            self.connections,
            self.syn_hyper.wmax_ee,
        )

        self.plotting_handler.plot_performance(
            self.exp_hyper.do_plot_performance,
            self.exp_hyper.num_examples,
            self.exp_hyper.update_interval,
        )

        for i, name in enumerate(self.net_hyper.input_population_names):
            self.input_groups[name + "e"].rates = 0 * b2.Hz

        variable_namespace = ModelEquations.create_variable_namespace(
            self.neuron_hyper, self.syn_hyper
        )

        net.run(0 * b2.second, namespace=variable_namespace)
        iteration: int = 0

        if self.exp_hyper.use_testing_set:
            input_data = self.testing_data["x"]
            target_data = self.testing_data["y"]
        else:
            input_data = self.training_data["x"]
            target_data = self.training_data["y"]

        while iteration < (int(self.exp_hyper.num_examples)):
            if not self.exp_hyper.test_mode:
                self.normalize_weights()

            data_index = iteration % input_data.shape[0]
            spike_rates = input_data[data_index, :, :].reshape((-1,)).astype(np.float64)
            spike_rates /= 8.0
            spike_rates *= self.net_hyper.input_intensity

            self.input_groups["Xe"].rates = spike_rates * b2.Hz

            net.run(
                self.exp_hyper.single_example_time,
                report="text",
                namespace=variable_namespace,
            )

            if iteration % self.exp_hyper.update_interval == 0 and iteration > 0:
                assignments = self.get_new_assignments(
                    self.result_monitor[:],
                    input_numbers[
                        iteration - self.exp_hyper.update_interval : iteration
                    ],
                )

            self.plotting_handler.update_input_weights_plot(
                iteration,
                self.exp_hyper.weight_update_interval,
                self.exp_hyper.test_mode,
                self.exp_hyper.n_input,
                self.exp_hyper.n_e,
                self.connections,
            )

            if (
                iteration % self.exp_hyper.save_connections_interval == 0
                and iteration > 0
                and not self.exp_hyper.test_mode
            ):
                save_connections(
                    self.net_hyper.save_conns,
                    self.connections,
                    self.exp_hyper.weight_path,
                    f"{self.exp_hyper.file_postfix}_{iteration}",
                )
                save_theta(
                    self.net_hyper.population_names,
                    self.exp_hyper.weight_path,
                    self.neuron_groups,
                    f"{self.exp_hyper.file_postfix}_{iteration}",
                )

            current_spike_count: np.ndarray = (
                np.asarray(self.spike_counters["Ae"].count[:]) - previous_spike_count
            )
            previous_spike_count: np.ndarray = np.copy(
                self.spike_counters["Ae"].count[:]
            )

            if np.sum(current_spike_count) < 5:
                # sample will be shown again with higher intensity
                self.net_hyper.input_intensity += 1
            else:
                self.result_monitor[
                    iteration % self.exp_hyper.update_interval, :
                ] = current_spike_count

                input_numbers[iteration] = target_data[data_index][0].item()

                output_numbers[iteration, :] = self.get_recognized_number_ranking(
                    assignments,
                    self.result_monitor[iteration % self.exp_hyper.update_interval, :],
                )

                if iteration % 100 == 0 and iteration > 0:
                    print(
                        f"runs done: {iteration} of {int(self.exp_hyper.num_examples)}",
                    )

                self.plotting_handler.update_performance_plot(
                    iteration,
                    self.exp_hyper.update_interval,
                    self.exp_hyper.do_plot_performance,
                    input_numbers,
                    output_numbers,
                )

                self.net_hyper.input_intensity = self.net_hyper.start_input_intensity
                iteration += 1

            for i, name in enumerate(self.net_hyper.input_population_names):
                self.input_groups[name + "e"].rates = 0 * b2.Hz

            net.run(
                self.exp_hyper.resting_time,
                namespace=variable_namespace,
            )

        self.save_results(input_numbers)

    def save_results(self, input_numbers: List[int]) -> None:
        """
        In training mode save the trained weights. Display the results in any case.

        :param input_numbers: target classes
        :return: None
        """

        print("save results")
        if not self.exp_hyper.test_mode:
            save_theta(
                self.net_hyper.population_names,
                self.exp_hyper.weight_path,
                self.neuron_groups,
                self.exp_hyper.file_postfix,
            )
        if not self.exp_hyper.test_mode:
            save_connections(
                self.net_hyper.save_conns,
                self.connections,
                self.exp_hyper.weight_path,
                self.exp_hyper.file_postfix,
            )
        else:
            np.save(
                self.exp_hyper.activity_path
                / f"resultPopVecs{self.exp_hyper.file_postfix}{self.exp_hyper.update_interval}",
                self.result_monitor,
            )
            np.save(
                self.exp_hyper.activity_path
                / f"inputNumbers{self.exp_hyper.file_postfix}{self.exp_hyper.update_interval}",
                input_numbers,
            )

        self.plotting_handler.plot_results(
            rate_monitors=self.rate_monitors,
            spike_monitors=self.spike_monitors,
            spike_counters=self.spike_counters,
            connections=self.connections,
            n_input=self.exp_hyper.n_input,
            n_e=self.exp_hyper.n_e,
            weight_max_ee=self.syn_hyper.wmax_ee,
        )


def main():
    Runner().run()


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import brian2 as b2
import numpy as np


@dataclass
class NeuronModelHyperparameters:
    v_rest_e: b2.Quantity
    """resting potential of an excitatory neuron"""
    v_rest_i: b2.Quantity
    """resting potential of an inhibitory neuron"""
    v_reset_e: b2.Quantity
    """potential after an excitatory neuron has spiked"""
    v_reset_i: b2.Quantity
    """potential after an inhibitory neuron has spiked"""
    v_thresh_e: b2.Quantity
    """threshold of an excitatory neuron"""
    v_thresh_i: b2.Quantity
    """threshold of an inhibitory neuron"""
    v_start_offset: b2.Quantity
    """offset that is applied to the reset potential at the start of the simulation"""
    v_equi_e_e: b2.Quantity
    """equilibrium potential of excitatory synapses of excitatory neurons"""
    v_equi_i_e: b2.Quantity
    """equilibrium potential of inhibitory synapses of excitatory neurons"""
    v_equi_e_i: b2.Quantity
    """equilibrium potential of excitatory synapses of inhibitory neurons"""
    v_equi_i_i: b2.Quantity
    """equilibrium potential of inhibitory synapses of inhibitory neurons"""
    tc_v_e: b2.Quantity
    """time constant for the membrane potential of excitatory neurons"""
    tc_v_i: b2.Quantity
    """time constant for the membrane potential of inhibitory neurons"""
    tc_ge: b2.Quantity
    """time constant for excitatory conductance"""
    tc_gi: b2.Quantity
    """time constant for inhibitory conductance"""
    refrac_e: b2.Quantity
    """refractory period duration of an excitatory neuron"""
    refrac_i: b2.Quantity
    """refractory period duration of an inhibitory neuron"""
    # TODO: why is this only used in the equation and not in the neuron group object?
    refrac_factor_e: float
    """factor applied to the refractory period when checking the threshold of an excitatory neuron"""

    @staticmethod
    def get_default() -> "NeuronModelHyperparameters":
        return NeuronModelHyperparameters(
            v_rest_e=-65.0 * b2.mV,
            v_rest_i=-60.0 * b2.mV,
            v_reset_e=-65.0 * b2.mV,
            v_reset_i=-45.0 * b2.mV,
            v_thresh_e=-52.0 * b2.mV,
            v_thresh_i=-40.0 * b2.mV,
            v_start_offset=-40.0 * b2.mV,
            v_equi_e_e=0.0 * b2.mV,
            v_equi_i_e=-100.0 * b2.mV,
            v_equi_e_i=0.0 * b2.mV,
            v_equi_i_i=-85.0 * b2.mV,
            tc_v_e=100.0 * b2.ms,
            tc_v_i=10.0 * b2.ms,
            tc_ge=1.0 * b2.ms,
            tc_gi=2.0 * b2.ms,
            refrac_e=5.0 * b2.ms,
            refrac_i=2.0 * b2.ms,
            refrac_factor_e=10.0,
        )


@dataclass
class SynapseModelHyperparameters:
    tc_pre_ee: b2.Quantity
    """time constant of the presynaptic trace of the learning rule"""
    tc_post_1_ee: b2.Quantity
    """time constant of the first postsynaptic trace of the learning rule"""
    tc_post_2_ee: b2.Quantity
    """time constant of the second postsynaptic trace of the learning rule"""
    nu_ee_pre: float
    """learning rate for presynaptic spikes"""
    nu_ee_post: float
    """learning rate for postsynaptic spikes"""
    wmax_ee: float
    """max weight"""
    theta_start: b2.Quantity
    """start value for the dynamic threshold theta"""
    tc_theta: b2.Quantity
    """time constant of theta"""
    theta_plus_e: b2.Quantity
    """value added to theta when neuron is reset"""
    offset: b2.Quantity
    """value subtracted from threshold"""

    @staticmethod
    def get_default() -> "SynapseModelHyperparameters":
        return SynapseModelHyperparameters(
            tc_pre_ee=20 * b2.ms,
            tc_post_1_ee=20 * b2.ms,
            tc_post_2_ee=40 * b2.ms,
            nu_ee_pre=0.0001,  # learning rate
            nu_ee_post=0.01,  # learning rate,
            wmax_ee=1.0,
            theta_start=20 * b2.mV,
            tc_theta=1e7 * b2.ms,
            theta_plus_e=0.05 * b2.mV,
            offset=20.0 * b2.mV,
        )


@dataclass
class ExperimentHyperparameters:
    test_mode: bool
    """determines test or training mode"""
    weight_path: Path
    """path to the folder where the weights should be saved in training mode"""
    activity_path: Path
    """path to the folder where the predictions should be saved in test mode"""
    num_examples: int
    """Number of examples that should be shown to the network. Values larger than the dataset can be used to train for multiple epochs."""
    use_testing_set: bool
    """determines if test or training data will be used"""
    do_plot_performance: bool
    """determines if the performance plot will be shown"""
    record_spikes: bool
    """determines if every single spike should be recorded"""
    ee_stdp_on: bool
    """determines whether the connection between input and excitatory neurons should be trained with STDP"""
    file_postfix: str
    """the postfix for the files that will be saved"""
    n_input: int
    """number of input values per sample"""
    n_e: int
    """number of excitatory neurons"""
    n_i: int
    """number of inhibitory neurons"""
    single_example_time: b2.Quantity
    """duration an input will be shown to the network"""
    resting_time: b2.Quantity
    """duration between two input examples where no input will be shown to the network"""
    runtime: b2.Quantity
    """estimation of the total runtime"""
    update_interval: int
    """number of iterations after which the assignments and performance plot get updated"""
    weight_update_interval: int
    """number of iterations after which the weights plot gets updated"""
    save_connections_interval: int
    """number of iterations after which the network parameters will be saved"""

    @staticmethod
    def get_default(test_mode: bool) -> "ExperimentHyperparameters":
        np.random.seed(0)
        weight_path = Path("weights")
        activity_path = Path("activity")

        if test_mode:
            num_examples = 10000 * 1
            use_testing_set = True
            do_plot_performance = False
            record_spikes = True
            ee_stdp_on = False
        else:
            num_examples = 60000 * 3
            use_testing_set = False
            do_plot_performance = True

            if num_examples <= 60000:
                record_spikes = True
            else:
                record_spikes = True

            ee_stdp_on = True

        file_postfix = ""
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
            weight_path=weight_path,
            activity_path=activity_path,
            num_examples=num_examples,
            use_testing_set=use_testing_set,
            do_plot_performance=do_plot_performance,
            record_spikes=record_spikes,
            ee_stdp_on=ee_stdp_on,
            update_interval=update_interval,
            file_postfix=file_postfix,
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
    """target weight sums for normalization"""
    delay: Dict[str, Tuple[b2.Quantity, b2.Quantity]]
    """delay ranges for input connections"""
    input_population_names: List[str]
    """names of the input neuron populations"""
    population_names: List[str]
    """names of the hidden neuron populations"""
    input_connection_names: List[str]
    """names of the connections between input neuron and hidden neuron populations"""
    save_conns: List[str]
    """names of the connections that should be saved to files"""
    input_conn_names: List[str]
    """input connection types"""
    recurrent_conn_names: List[str]
    """recurrent connection types"""
    input_intensity: float
    """multiplicative factor for the input values"""
    start_input_intensity: float
    """start value of the input intensity"""

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
    """equation for resetting the neuron"""
    v_thresh_e_eqs: str
    """equation for checking the threshold of excitatory neurons"""
    v_thresh_i_eqs: str
    """equation for checking the threshold of inhibitory neurons"""
    v_reset_i_eqs: str
    """equation for resetting the inhibitory neuron"""
    neuron_eqs_e: str
    """equation defining the membrane potential changes for excitatory neurons"""
    neuron_eqs_i: str
    """equation defining the membrane potential changes for inhibitory neurons"""
    syn_eqs: str
    """equation setting up the synapses"""
    syn_eqs_pre_e: str
    """equation defining the behavior of an excitatory synapse during a presynaptic spike"""
    syn_eqs_pre_i: str
    """equation defining the behavior of an inhibitory synapse during a presynaptic spike"""
    syn_eqs_post: str
    """equation defining the behavior of a synapse during a postsynaptic spike"""
    eqs_stdp_ee: str
    """equation defining the exponential decay of the traces of the learning rule"""
    eqs_stdp_pre_ee: str
    """equation defining the learning rule for presynaptic spikes"""
    eqs_stdp_post_ee: str
    """equation defining the learning rule for postsynaptic spikes"""

    @staticmethod
    def get_default(test_mode: bool) -> "ModelEquations":
        if test_mode:
            scr_e = "v = v_reset_e; timer = 0*ms"
        else:
            scr_e = "v = v_reset_e; theta += theta_plus_e; timer = 0*ms"

        v_thresh_e_eqs = (
            "(v>(theta - offset + v_thresh_e)) and (timer>(refrac_e * refrac_factor_e))"
        )
        v_thresh_i_eqs = "v>v_thresh_i"
        v_reset_i_eqs = "v=v_reset_i"

        neuron_eqs_e = """
           dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / tc_v_e    : volt (unless refractory)
           I_synE = ge * nS * (v_equi_e_e - v)                         : amp
           I_synI = gi * nS * (v_equi_i_e - v)                         : amp
           dge/dt = -ge/tc_ge                                          : 1
           dgi/dt = -gi/tc_gi                                          : 1
           """

        if test_mode:
            neuron_eqs_e += "\n  theta      :volt"
        else:
            neuron_eqs_e += "\n  dtheta/dt = -theta / (tc_theta)  : volt"

        neuron_eqs_e += "\n  dtimer/dt = 1.0  : second"

        neuron_eqs_i = """
            dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / tc_v_i   : volt (unless refractory)
            I_synE = ge * nS * (v_equi_e_i - v)                        : amp
            I_synI = gi * nS * (v_equi_i_i - v)                        : amp
            dge/dt = -ge/tc_ge                                         : 1
            dgi/dt = -gi/tc_gi                                         : 1
            """

        syn_eqs = "w : 1"
        syn_eqs_pre_e = "ge_post += w"  # _post means it uses this variable from the postsynaptic neuron
        syn_eqs_pre_i = "gi_post += w"  # _post means it uses this variable from the postsynaptic neuron
        syn_eqs_post = ""

        eqs_stdp_ee = """
            post2before                            : 1
            dpre/dt    = -pre/(tc_pre_ee)          : 1 (event-driven)
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
            syn_eqs=syn_eqs,
            syn_eqs_pre_e=syn_eqs_pre_e,
            syn_eqs_pre_i=syn_eqs_pre_i,
            syn_eqs_post=syn_eqs_post,
            eqs_stdp_ee=eqs_stdp_ee,
            eqs_stdp_pre_ee=eqs_stdp_pre_ee,
            eqs_stdp_post_ee=eqs_stdp_post_ee,
        )

    @staticmethod
    def create_variable_namespace(
        neuron_model_hyperparameters: NeuronModelHyperparameters,
        synapse_model_hyperparameters: SynapseModelHyperparameters,
    ):
        return {
            "v_rest_e": neuron_model_hyperparameters.v_rest_e,
            "v_rest_i": neuron_model_hyperparameters.v_rest_i,
            "v_thresh_e": neuron_model_hyperparameters.v_thresh_e,
            "v_thresh_i": neuron_model_hyperparameters.v_thresh_i,
            "refrac_e": neuron_model_hyperparameters.refrac_e,
            "refrac_factor_e": neuron_model_hyperparameters.refrac_factor_e,
            "v_reset_e": neuron_model_hyperparameters.v_rest_e,
            "v_reset_i": neuron_model_hyperparameters.v_reset_i,
            "v_equi_e_e": neuron_model_hyperparameters.v_equi_e_e,
            "v_equi_i_e": neuron_model_hyperparameters.v_equi_i_e,
            "v_equi_e_i": neuron_model_hyperparameters.v_equi_e_i,
            "v_equi_i_i": neuron_model_hyperparameters.v_equi_i_i,
            "tc_v_e": neuron_model_hyperparameters.tc_v_e,
            "tc_v_i": neuron_model_hyperparameters.tc_v_i,
            "tc_ge": neuron_model_hyperparameters.tc_ge,
            "tc_gi": neuron_model_hyperparameters.tc_gi,
            "nu_ee_pre": synapse_model_hyperparameters.nu_ee_pre,
            "tc_post_1_ee": synapse_model_hyperparameters.tc_post_1_ee,
            "tc_post_2_ee": synapse_model_hyperparameters.tc_post_2_ee,
            "tc_pre_ee": synapse_model_hyperparameters.tc_pre_ee,
            "wmax_ee": synapse_model_hyperparameters.wmax_ee,
            "nu_ee_post": synapse_model_hyperparameters.nu_ee_post,
            "offset": synapse_model_hyperparameters.offset,
            "tc_theta": synapse_model_hyperparameters.tc_theta,
            "theta_plus_e": synapse_model_hyperparameters.theta_plus_e,
        }

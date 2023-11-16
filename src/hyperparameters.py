from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import brian2 as b2
import numpy as np


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
            num_examples = 60000 * 3
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

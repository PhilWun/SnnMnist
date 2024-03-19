import brian2 as b2
import mlflow
import numpy as np
from brian2 import SpikeGeneratorGroup

from src.hyperparameters import (
    NeuronModelHyperparameters,
    ModelEquations,
    SynapseModelHyperparameters,
)


def main():
    test_mode = False
    neuron_hyper = NeuronModelHyperparameters.get_default()
    model_equations = ModelEquations.get_default(test_mode=test_mode)
    syn_hyper = SynapseModelHyperparameters.get_default()

    input_generator = SpikeGeneratorGroup(
        2,
        b2.array([0, 1] + [0] * 110),
        b2.array([5, 60] + list(range(70, 70 + 110))) * b2.ms,
    )
    neurons = b2.NeuronGroup(
        1,
        model_equations.neuron_eqs_e,
        threshold=model_equations.v_thresh_e_eqs,
        refractory=neuron_hyper.refrac_e,
        reset=model_equations.scr_e,
        method="euler",
    )

    if test_mode:
        excitatory_synapses = b2.Synapses(
            input_generator,
            neurons,
            model=model_equations.syn_eqs,
            on_pre=model_equations.syn_eqs_pre_e,
            on_post=model_equations.syn_eqs_post,
            method="euler",
        )
    else:
        excitatory_synapses = b2.Synapses(
            input_generator,
            neurons,
            model=model_equations.syn_eqs + model_equations.eqs_stdp_ee,
            on_pre=model_equations.syn_eqs_pre_e
            + "; "
            + model_equations.eqs_stdp_pre_ee,
            on_post=model_equations.syn_eqs_post + model_equations.eqs_stdp_post_ee,
            method="euler",
        )

    inhibitory_synapses = b2.Synapses(
        input_generator,
        neurons,
        model=model_equations.syn_eqs,
        on_pre=model_equations.syn_eqs_pre_i,
        on_post=model_equations.syn_eqs_post,
    )

    excitatory_synapses.connect(condition=True)
    inhibitory_synapses.connect(condition=True)

    weights_e = np.array([[0.8], [0.0]])
    weights_i = np.array([[0.0], [10.0]])

    excitatory_synapses.w = weights_e[excitatory_synapses.i, excitatory_synapses.j]
    inhibitory_synapses.w = weights_i[inhibitory_synapses.i, inhibitory_synapses.j]

    neuron_monitor = b2.StateMonitor(neurons, ["v", "theta", "ge", "gi"], record=True)
    synapse_monitor = b2.StateMonitor(
        excitatory_synapses, ["w", "post2before", "pre", "post1", "post2"], record=True
    )
    variable_namespace = ModelEquations.create_variable_namespace(
        neuron_hyper, syn_hyper
    )

    neurons.v = neuron_hyper.v_rest_e + neuron_hyper.v_start_offset
    neurons.theta = neuron_hyper.theta_start

    b2.run(200 * b2.ms, namespace=variable_namespace)

    # b2.plot(neuron_monitor.t / b2.ms, neuron_monitor.v[0] / b2.mV)
    # b2.xlabel("Time (ms)")
    # b2.ylabel("v (mV)")
    # b2.show()
    #
    # b2.plot(neuron_monitor.t / b2.ms, neuron_monitor.theta[0] / b2.mV)
    # b2.xlabel("Time (ms)")
    # b2.ylabel("theta (mV)")
    # b2.show()
    #
    # b2.plot(neuron_monitor.t / b2.ms, neuron_monitor.ge[0])
    # b2.xlabel("Time (ms)")
    # b2.ylabel("ge")
    # b2.show()
    #
    # b2.plot(neuron_monitor.t / b2.ms, neuron_monitor.gi[0])
    # b2.xlabel("Time (ms)")
    # b2.ylabel("gi")
    # b2.show()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.start_run(experiment_id="9", tags={"implementation": "python"})

    for i, v in enumerate(neuron_monitor.v[0]):
        mlflow.log_metric("neuron_v", v, step=i)

    for i, theta in enumerate(neuron_monitor.theta[0]):
        mlflow.log_metric("neuron_theta", theta, step=i)

    for i, ge in enumerate(neuron_monitor.ge[0]):
        mlflow.log_metric("neuron_ge", ge, step=i)

    for i, gi in enumerate(neuron_monitor.gi[0]):
        mlflow.log_metric("neuron_gi", gi, step=i)

    for i, w in enumerate(synapse_monitor.w[0]):
        mlflow.log_metric("exc_syn_w", w, step=i)

    for i, post2before in enumerate(synapse_monitor.post2before[0]):
        mlflow.log_metric("exc_syn_post2before", post2before, step=i)

    for i, pre in enumerate(synapse_monitor.pre[0]):
        mlflow.log_metric("exc_syn_pre", pre, step=i)

    for i, post1 in enumerate(synapse_monitor.post1[0]):
        mlflow.log_metric("exc_syn_post1", post1, step=i)

    for i, post2 in enumerate(synapse_monitor.post2[0]):
        mlflow.log_metric("exc_syn_post2", post2, step=i)

    mlflow.end_run()


if __name__ == "__main__":
    main()

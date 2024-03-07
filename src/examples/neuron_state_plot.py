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
    neuron_hyper = NeuronModelHyperparameters.get_default()
    model_equations = ModelEquations.get_default(test_mode=False)
    syn_hyper = SynapseModelHyperparameters.get_default()

    input_generator = SpikeGeneratorGroup(
        2, b2.array([0, 1]), b2.array([5, 10]) * b2.ms
    )
    neurons = b2.NeuronGroup(
        1,
        model_equations.neuron_eqs_e,
        threshold=model_equations.v_thresh_e_eqs,
        refractory=neuron_hyper.refrac_e,
        reset=model_equations.scr_e,
        method="euler",
    )
    excitatory_synapses = b2.Synapses(
        input_generator,
        neurons,
        model=model_equations.syn_eqs,
        on_pre=model_equations.syn_eqs_pre_e,
        on_post=model_equations.syn_eqs_post,
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

    weights_e = np.array([[40.0], [0.0]])
    weights_i = np.array([[0.0], [30.0]])

    excitatory_synapses.w = weights_e[excitatory_synapses.i, excitatory_synapses.j]
    inhibitory_synapses.w = weights_i[inhibitory_synapses.i, inhibitory_synapses.j]

    monitor = b2.StateMonitor(neurons, ["v", "theta", "ge", "gi"], record=True)
    variable_namespace = ModelEquations.create_variable_namespace(
        neuron_hyper, syn_hyper
    )

    neurons.v = neuron_hyper.v_rest_e + neuron_hyper.v_start_offset
    neurons.theta = neuron_hyper.theta_start

    b2.run(100 * b2.ms, namespace=variable_namespace)

    # b2.plot(monitor.t / b2.ms, monitor.v[0] / b2.mV)
    # b2.xlabel("Time (ms)")
    # b2.ylabel("v (mV)")
    # b2.show()
    #
    # b2.plot(monitor.t / b2.ms, monitor.theta[0] / b2.mV)
    # b2.xlabel("Time (ms)")
    # b2.ylabel("theta (mV)")
    # b2.show()
    #
    # b2.plot(monitor.t / b2.ms, monitor.ge[0])
    # b2.xlabel("Time (ms)")
    # b2.ylabel("ge")
    # b2.show()
    #
    # b2.plot(monitor.t / b2.ms, monitor.gi[0])
    # b2.xlabel("Time (ms)")
    # b2.ylabel("gi")
    # b2.show()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.start_run(experiment_id="8", tags={"implementation": "python"})

    for i, v in enumerate(monitor.v[0]):
        mlflow.log_metric("v", v, step=i)

    for i, theta in enumerate(monitor.theta[0]):
        mlflow.log_metric("theta", theta, step=i)

    for i, ge in enumerate(monitor.ge[0]):
        mlflow.log_metric("ge", ge, step=i)

    for i, gi in enumerate(monitor.gi[0]):
        mlflow.log_metric("gi", gi, step=i)

    mlflow.end_run()


if __name__ == "__main__":
    main()

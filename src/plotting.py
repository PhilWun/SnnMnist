from typing import Dict, List, Tuple

import brian2 as b2
import matplotlib
import numpy as np
from brian2tools import brian_plot
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D


def get_2d_input_weights(
    n_input: int, n_e: int, connections: Dict[str, b2.Synapses]
) -> np.ndarray:
    name = "XeAe"
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    conn_matrix = np.zeros((n_input, n_e))
    conn_matrix[connections[name].i, connections[name].j] = connections[name].w
    weight_matrix = np.copy(conn_matrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
            rearranged_weights[
                i * n_in_sqrt : (i + 1) * n_in_sqrt, j * n_in_sqrt : (j + 1) * n_in_sqrt
            ] = weight_matrix[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))

    return rearranged_weights


def plot_2d_input_weights(
    n_input: int,
    n_e: int,
    connections: Dict[str, b2.Synapses],
    fig_num: int,
    weight_max_ee: float,
) -> Tuple[AxesImage, b2.Figure]:
    name = "XeAe"
    weights = get_2d_input_weights(n_input, n_e, connections)
    fig = b2.figure(fig_num, figsize=(18, 18))
    im2 = b2.imshow(
        weights,
        interpolation="nearest",
        vmin=0,
        vmax=weight_max_ee,
        cmap=matplotlib.colormaps.get_cmap("hot_r"),
    )
    b2.colorbar(im2)
    b2.title("weights of connection" + name)
    fig.canvas.draw()

    return im2, fig


def update_2d_input_weights(
    im: AxesImage,
    fig: b2.Figure,
    n_input: int,
    n_e: int,
    connections: Dict[str, b2.Synapses],
) -> AxesImage:
    weights = get_2d_input_weights(n_input, n_e, connections)
    im.set_array(weights)
    fig.canvas.draw()

    return im


def get_current_performance(
    performance: np.ndarray,
    current_example_num: int,
    update_interval: int,
    output_numbers: np.ndarray,
    input_numbers: List[int],
) -> np.ndarray:
    current_evaluation = int(current_example_num / update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = output_numbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100

    return performance


def plot_performance(
    fig_num: int, num_examples: int, update_interval: int
) -> Tuple[Line2D, np.ndarray, int, b2.Figure]:
    num_evaluations = int(num_examples / update_interval)
    time_steps = range(0, num_evaluations)
    performance: np.ndarray = np.zeros(num_evaluations)
    fig: b2.Figure = b2.figure(fig_num, figsize=(5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2: Line2D = ax.plot(time_steps, performance)[0]  # my_cmap
    b2.ylim(ymax=100)
    b2.title("Classification performance")
    fig.canvas.draw()

    return im2, performance, fig_num, fig


def update_performance_plot(
    im: Line2D,
    performance: np.ndarray,
    current_example_num: int,
    fig: b2.Figure,
    update_interval: int,
    output_numbers: np.ndarray,
    input_numbers: List[int],
) -> Tuple[Line2D, np.ndarray]:
    performance = get_current_performance(
        performance, current_example_num, update_interval, output_numbers, input_numbers
    )
    im.set_ydata(performance)
    fig.canvas.draw()

    return im, performance


def plot_results(
    fig_num: int,
    rate_monitors: Dict[str, b2.PopulationRateMonitor],
    spike_monitors: Dict[str, b2.SpikeMonitor],
    spike_counters: Dict[str, b2.SpikeMonitor],
    connections: Dict[str, b2.Synapses],
    n_input: int,
    n_e: int,
    weight_max_ee: float,
):
    # ------------------------------------------------------------------------------
    # plot results
    # ------------------------------------------------------------------------------
    if rate_monitors:
        b2.figure(fig_num)
        fig_num += 1

        for i, name in enumerate(rate_monitors):
            b2.subplot(len(rate_monitors), 1, 1 + i)
            b2.plot(
                rate_monitors[name].t / b2.second,
                rate_monitors[name].rate,
                ".",
            )
            b2.title("Rates of population " + name)

    if spike_monitors:
        b2.figure(fig_num)
        fig_num += 1

        for i, name in enumerate(spike_monitors):
            b2.subplot(len(spike_monitors), 1, 1 + i)
            b2.plot(
                spike_monitors[name].t / b2.ms,
                spike_monitors[name].i,
                ".",
            )
            b2.title("Spikes of population " + name)

    if spike_counters:
        b2.figure(fig_num)
        fig_num += 1
        b2.plot(spike_monitors["Ae"].count[:])
        b2.title("Spike count of population Ae")

    plot_2d_input_weights(n_input, n_e, connections, fig_num, weight_max_ee)

    b2.plt.figure(5)

    b2.subplot(3, 1, 1)

    brian_plot(connections["XeAe"].w)
    b2.subplot(3, 1, 2)

    brian_plot(connections["AeAi"].w)

    b2.subplot(3, 1, 3)

    brian_plot(connections["AiAe"].w)

    b2.plt.figure(6)

    b2.subplot(3, 1, 1)

    brian_plot(connections["XeAe"].delay)
    b2.subplot(3, 1, 2)

    brian_plot(connections["AeAi"].delay)

    b2.subplot(3, 1, 3)

    brian_plot(connections["AiAe"].delay)

    b2.ioff()
    b2.show()

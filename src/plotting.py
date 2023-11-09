from typing import Dict, List, Any, Tuple

import brian2 as b2
import matplotlib
import numpy as np


def get_2d_input_weights(n_input: int, n_e: int, connections: Dict[str, b2.Synapses]):
    name = "XeAe"
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = np.zeros((n_input, n_e))
    connMatrix[connections[name].i, connections[name].j] = connections[name].w
    weight_matrix = np.copy(connMatrix)

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
    wmax_ee: float,
):
    name = "XeAe"
    weights = get_2d_input_weights(n_input, n_e, connections)
    fig = b2.figure(fig_num, figsize=(18, 18))
    im2 = b2.imshow(
        weights,
        interpolation="nearest",
        vmin=0,
        vmax=wmax_ee,
        cmap=matplotlib.colormaps.get_cmap("hot_r"),
    )
    b2.colorbar(im2)
    b2.title("weights of connection" + name)
    fig.canvas.draw()

    return im2, fig


def update_2d_input_weights(
    im, fig, n_input: int, n_e: int, connections: Dict[str, b2.Synapses]
):
    weights = get_2d_input_weights(n_input, n_e, connections)
    im.set_array(weights)
    fig.canvas.draw()
    return im


def get_current_performance(
    performance,
    current_example_num,
    update_interval: int,
    outputNumbers: np.ndarray,
    input_numbers: List[int],
):
    current_evaluation = int(current_example_num / update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_performance(
    fig_num: int, num_examples: int, update_interval: int
) -> Tuple[Any, np.ndarray, int, b2.Figure]:
    num_evaluations = int(num_examples / update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b2.figure(fig_num, figsize=(5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    (im2,) = ax.plot(time_steps, performance)  # my_cmap
    b2.ylim(ymax=100)
    b2.title("Classification performance")
    fig.canvas.draw()

    return im2, performance, fig_num, fig


def update_performance_plot(
    im,
    performance,
    current_example_num,
    fig,
    update_interval: int,
    outputNumbers: np.ndarray,
    input_numbers: List[int],
):
    performance = get_current_performance(
        performance, current_example_num, update_interval, outputNumbers, input_numbers
    )
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

from typing import Dict, List, Tuple

import brian2 as b2
import matplotlib
import numpy as np
from brian2tools import brian_plot
from matplotlib.figure import Figure
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


def add_current_performance(
    performance: List[float],
    current_example_num: int,
    update_interval: int,
    output_numbers: np.ndarray,
    input_numbers: List[int],
) -> None:
    current_evaluation = int(current_example_num / update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = output_numbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance.append(correct / float(update_interval) * 100)


def plot_performance(
    fig_num: int, num_examples: int, update_interval: int
) -> Tuple[Line2D, List[float], int, b2.Figure]:
    time_steps = range(0, 1)
    performance = [0.0]
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
    performance: List[float],
    current_example_num: int,
    fig: b2.Figure,
    update_interval: int,
    output_numbers: np.ndarray,
    input_numbers: List[int],
) -> Line2D:
    add_current_performance(
        performance, current_example_num, update_interval, output_numbers, input_numbers
    )
    im.set_ydata(performance)
    im.set_xdata(range(0, len(performance)))
    fig.axes[0].relim()
    fig.axes[0].autoscale_view(True, True, True)
    fig.canvas.draw()

    return im


class PlottingHandler:
    def __init__(self):
        b2.ion()
        self.fig_num = 1
        self.input_weight_monitor: AxesImage | None = None
        self.fig_weights: Figure | None = None
        self.performance_monitor: Line2D | None = None
        self.performance: List[float] = []
        self.fig_performance: b2.Figure | None = None

    def plot_input_weights(
        self,
        test_mode: bool,
        n_input: int,
        n_e: int,
        connections: Dict[str, b2.Synapses],
        wmax_ee: float,
    ):
        if test_mode:
            return

        self.input_weight_monitor, self.fig_weights = plot_2d_input_weights(
            n_input,
            n_e,
            connections,
            self.fig_num,
            wmax_ee,
        )
        self.fig_num += 1

    def update_input_weights_plot(
        self,
        iteration: int,
        weight_update_interval: int,
        test_mode: bool,
        n_input: int,
        n_e: int,
        connections: Dict[str, b2.Synapses],
    ):
        if test_mode:
            return

        if iteration % weight_update_interval != 0:
            return

        update_2d_input_weights(
            self.input_weight_monitor,
            self.fig_weights,
            n_input,
            n_e,
            connections,
        )
        b2.pause(0.1)  # triggers update of the plots

    def plot_performance(
        self, do_plot_performance: bool, num_examples: int, update_interval: int
    ):
        if not do_plot_performance:
            return

        (
            self.performance_monitor,
            self.performance,
            self.fig_num,
            self.fig_performance,
        ) = plot_performance(
            self.fig_num,
            num_examples,
            update_interval,
        )

    def update_performance_plot(
        self,
        iteration: int,
        update_interval: int,
        do_plot_performance: bool,
        input_numbers: List[int],
        output_numbers: np.ndarray,
    ):
        should_update = iteration % update_interval == 0 and iteration > 0

        if not should_update:
            return

        if not do_plot_performance:
            return

        update_performance_plot(
            self.performance_monitor,
            self.performance,
            iteration,
            self.fig_performance,
            update_interval,
            output_numbers,
            input_numbers,
        )

        index = int(iteration / float(update_interval))

        print(
            "Classification performance",
            self.performance[: index + 1],
        )

    def plot_results(
        self,
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
            b2.figure(self.fig_num)
            self.fig_num += 1

            for i, name in enumerate(rate_monitors):
                b2.subplot(len(rate_monitors), 1, 1 + i)
                b2.plot(
                    rate_monitors[name].t / b2.second,
                    rate_monitors[name].rate,
                    ".",
                )
                b2.title("Rates of population " + name)

        if spike_monitors:
            b2.figure(self.fig_num)
            self.fig_num += 1

            for i, name in enumerate(spike_monitors):
                b2.subplot(len(spike_monitors), 1, 1 + i)
                b2.plot(
                    spike_monitors[name].t / b2.ms,
                    spike_monitors[name].i,
                    ".",
                )
                b2.title("Spikes of population " + name)

        if spike_counters:
            b2.figure(self.fig_num)
            self.fig_num += 1
            b2.plot(spike_monitors["Ae"].count[:])
            b2.title("Spike count of population Ae")

        plot_2d_input_weights(n_input, n_e, connections, self.fig_num, weight_max_ee)

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

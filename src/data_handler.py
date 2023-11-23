import os.path
import pickle
from pathlib import Path

from struct import unpack
from typing import Dict, TypedDict, List

import brian2 as b2
import numpy as np

# specify the location of the MNIST data
MNIST_data_path = Path("MNIST")


class LabeledData(TypedDict):
    x: np.ndarray
    y: np.ndarray
    rows: int
    cols: int


def get_labeled_data(pickle_file_name: str, training=True) -> LabeledData:
    """
    Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    """
    if os.path.isfile(f"{MNIST_data_path / pickle_file_name}.pickle"):
        data = pickle.load(
            open(f"{MNIST_data_path / pickle_file_name}.pickle", mode="rb")
        )
    else:
        # Open the images with gzip in read binary mode
        if training:
            images = open(MNIST_data_path / "train-images-idx3-ubyte", "rb")
            labels = open(MNIST_data_path / "train-labels-idx1-ubyte", "rb")
        else:
            images = open(MNIST_data_path / "t10k-images-idx3-ubyte", "rb")
            labels = open(MNIST_data_path / "t10k-labels-idx1-ubyte", "rb")

        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images: int = unpack(">I", images.read(4))[0]
        rows: int = unpack(">I", images.read(4))[0]
        cols: int = unpack(">I", images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        number_of_labels: int = unpack(">I", labels.read(4))[0]

        if number_of_images != number_of_labels:
            raise Exception("number of labels did not match the number of images")
        # Get the data
        x: np.ndarray = np.zeros(
            (number_of_labels, rows, cols), dtype=np.uint8
        )  # Initialize numpy array
        y: np.ndarray = np.zeros(
            (number_of_labels, 1), dtype=np.uint8
        )  # Initialize numpy array

        for i in range(number_of_labels):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [
                [unpack(">B", images.read(1))[0] for unused_col in range(cols)]
                for unused_row in range(rows)
            ]
            y[i] = unpack(">B", labels.read(1))[0]

        data: LabeledData = {"x": x, "y": y, "rows": rows, "cols": cols}
        pickle.dump(data, open(f"{MNIST_data_path / pickle_file_name}.pickle", "wb"))

    return data


def get_matrix_from_file(
    file_name: Path, ending: str, n_input: int, n_e: int, n_i: int
) -> np.ndarray:
    offset = len(ending) + 4

    if file_name.name[-4 - offset] == "X":
        n_src = n_input
    else:
        if file_name.name[-3 - offset] == "e":
            n_src = n_e
        else:
            n_src = n_i

    if file_name.name[-1 - offset] == "e":
        n_tgt = n_e
    else:
        n_tgt = n_i

    readout: np.ndarray = np.load(file_name)
    print(readout.shape, file_name)
    value_arr: np.ndarray = np.zeros((n_src, n_tgt))

    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]

    return value_arr


def save_connections(
    save_conns: List[str],
    connections: Dict[str, b2.Synapses],
    weight_path: Path,
    ending="",
) -> None:
    print("save connections")

    for connName in save_conns:
        conn = connections[connName]
        conn_list_sparse = np.stack([conn.i, conn.j, conn.w], axis=1)
        np.save(weight_path / f"{connName}{ending}", conn_list_sparse)


def save_theta(
    population_names: List[str],
    weight_path: Path,
    neuron_groups: Dict[str, b2.NeuronGroup],
    ending="",
) -> None:
    print("save theta")

    for pop_name in population_names:
        np.save(
            weight_path / f"theta_{pop_name}{ending}",
            neuron_groups[pop_name + "e"].theta,
        )

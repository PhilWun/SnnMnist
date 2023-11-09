import os.path
import pickle
from pathlib import Path

from struct import unpack
import numpy as np

# specify the location of the MNIST data
MNIST_data_path = Path("MNIST")


def get_labeled_data(picklename, bTrain=True):
    """Read input-vector (image) and target class (label, 0-9) and return
    it as list of tuples.
    """
    if os.path.isfile(f"{MNIST_data_path / picklename}.pickle"):
        data = pickle.load(open(f"{MNIST_data_path / picklename}.pickle", mode="rb"))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path / "train-images-idx3-ubyte", "rb")
            labels = open(MNIST_data_path / "train-labels-idx1-ubyte", "rb")
        else:
            images = open(MNIST_data_path / "t10k-images-idx3-ubyte", "rb")
            labels = open(MNIST_data_path / "t10k-labels-idx1-ubyte", "rb")
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack(">I", images.read(4))[0]
        rows = unpack(">I", images.read(4))[0]
        cols = unpack(">I", images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack(">I", labels.read(4))[0]

        if number_of_images != N:
            raise Exception("number of labels did not match the number of images")
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [
                [unpack(">B", images.read(1))[0] for unused_col in range(cols)]
                for unused_row in range(rows)
            ]
            y[i] = unpack(">B", labels.read(1))[0]

        data = {"x": x, "y": y, "rows": rows, "cols": cols}
        pickle.dump(data, open(f"{MNIST_data_path / picklename}.pickle", "wb"))

    return data


def get_matrix_from_file(fileName, ending: str, n_input: int, n_e: int, n_i: int):
    offset = len(ending) + 4
    if fileName[-4 - offset] == "X":
        n_src = n_input
    else:
        if fileName[-3 - offset] == "e":
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1 - offset] == "e":
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]
    return value_arr


def save_connections(save_conns, connections, data_path, ending=""):
    print("save connections")
    for connName in save_conns:
        conn = connections[connName]
        connListSparse = np.stack([conn.i, conn.j, conn.w], axis=1)
        np.save(data_path + "weights/" + connName + ending, connListSparse)


def save_theta(population_names, data_path, neuron_groups, ending=""):
    print("save theta")
    for pop_name in population_names:
        np.save(
            data_path + "weights/theta_" + pop_name + ending,
            neuron_groups[pop_name + "e"].theta,
        )

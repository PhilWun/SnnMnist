"""
Created on 15.12.2014

@author: Peter U. Diehl
"""
from pathlib import Path
from typing import List

import numpy as np
import brian2 as b2

from src.experiment import Runner


def main():
    data_path = Path("activity")
    num_time_steps = 10000
    """equals the number of examples shown during training / testing"""
    start_evaluation_time = 0
    end_evaluation_time = num_time_steps
    num_segments = 10
    time_steps_per_segment = num_time_steps // num_segments

    print("load results")
    result_monitor = np.load(data_path / f"resultPopVecs{num_time_steps}.npy")
    input_numbers = np.load(data_path / f"inputNumbers{num_time_steps}.npy")
    print(result_monitor.shape)

    runner = Runner()

    print("get assignments")
    assignments = runner.get_new_assignments(
        result_monitor[start_evaluation_time:end_evaluation_time],
        input_numbers[start_evaluation_time:end_evaluation_time],
    )
    print(assignments)

    sum_accuracy: List[float] = [0.0] * num_segments

    for current_segment in range(num_segments):
        end_time = min(
            end_evaluation_time, (current_segment + 1) * time_steps_per_segment
        )
        start_time = time_steps_per_segment * current_segment
        test_results = np.zeros((10, end_time - start_time))
        print("calculate accuracy for sum")

        for i in range(end_time - start_time):
            test_results[:, i] = runner.get_recognized_number_ranking(
                assignments, result_monitor[i + start_time, :]
            )

        difference = test_results[0, :] - input_numbers[start_time:end_time]
        correct = len(np.where(difference == 0)[0])
        incorrect = np.where(difference != 0)[0]
        sum_accuracy[current_segment] = correct / float(end_time - start_time) * 100
        print(
            "Sum response - accuracy: ",
            sum_accuracy[current_segment],
            " num incorrect: ",
            len(incorrect),
        )

    print(
        "Sum response - accuracy --> mean: ",
        np.mean(sum_accuracy),
        "--> standard deviation: ",
        np.std(sum_accuracy),
    )

    b2.show()


if __name__ == "__main__":
    main()

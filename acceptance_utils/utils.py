import time
import numpy as np
import matplotlib.pyplot as plt


def estimate_statistics(input_state, theta_list, n_samples, overhead_time):
    """This function runs an experiment and
    returns the number of samples effectively collected, the rates, the accuracy,
    the time taken to collect the samples, and the samples themselves"""

    start_time = time.time()
    samples = tbi.sample(
        input_state=input_state, theta_list=theta_list, n_samples=n_samples
    )
    request_time = max([time.time() - start_time - overhead_time_with_margin, 0.1])

    rates = n_samples / request_time
    target_state = samples.get(input_state, 0)
    accuracy = 100 - 100 * (n_samples - target_state) / n_samples

    return n_samples, rates, accuracy, request_time, samples


def plot(
    samples,
    show_plot=False,
    title="Counts Data",
):
    ## Create a bar plot
    keys = ["".join(map(str, key)) for key in samples.keys()]
    values = list(samples.values())
    fig = plt.figure(figsize=(6, 4), dpi=200)
    plt.bar(keys, values)
    plt.xticks(rotation=90, fontsize=4)
    # Add labels and title
    plt.xlabel("Output state")
    plt.ylabel("Occurences")
    plt.title(f"{title}")
    ## Show the plot
    if show_plot:
        plt.show()

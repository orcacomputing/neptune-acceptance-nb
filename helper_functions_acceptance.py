import time
import numpy as np
import ptseries
from ptseries.tbi import create_tbi
# from uncertainties import ufloat
from math import factorial
# import uncertainties
# import uncertainties.umath as um
from scipy.special import comb
import numpy as np
import scipy.special as sc
import math
import matplotlib.pyplot as plt
from datetime import datetime as datetime

N_DET = 6

def get_date_as_string(format="%Y-%m-%d_%H-%M-%S"):
    # Create a datetime object representing the current date and time
    current_datetime = datetime.now()
    # Format the datetime object into a string with YYYY-MM-DD format
    date_string = current_datetime.strftime(format)
    return date_string

################################################################
############### FROM CLICKS TO PROBS FUNCS #####################
################################################################
def DET(mMax, nMax, eta):
    det = np.zeros((mMax + 1, nMax + 1))

    for m in range(mMax + 1):
        for nn in range(nMax + 1):
            if m > nn:
                det[m, nn] = 0
            elif m < nn:
                summary = []
                for j in range(0, m + 1):
                    term = (
                        (-1) ** j
                        * sc.binom(m, j)
                        * ((1 - eta) + ((m - j) * eta / mMax)) ** nn
                    )
                    summary.append(term)
                det[m, nn] = sc.binom(mMax, m) * np.sum(summary)
            else:
                det[m, nn] = (eta / mMax) ** nn * (
                    factorial(mMax) / factorial(mMax - nn)
                )

    return det


def EME(nMax, det, l, c):
    iterations = 10**10
    epsilon = 10 ** (-12)

    pn = np.array([1.0 / (nMax + 1)] * (nMax + 1))
    iteration = 0

    while iteration < iterations:
        EM = np.dot(c / np.dot(det, pn), det) * pn
        E = (
            l
            * (
                np.array([um.log(x) for x in pn])
                - np.sum(pn * np.array([um.log(x) for x in pn]))
            )
            * pn
        )
        # E = l * (np.log(pn) - np.sum(pn * np.log(pn))) * pn
        if isinstance(E[0], uncertainties.UFloat):
            E[[np.isnan(x.nominal_value) for x in E]] = 0.0
        else:
            E[np.isnan(E)] = 0.0

        EME_eval = EM - E
        dist = um.sqrt(np.sum((EME_eval - pn) ** 2))
        if dist <= epsilon:
            break
        else:
            pn = EME_eval

        iteration += 1

    return EME_eval


def construct_C_matrix(M, eta=1):
    C = np.zeros((M + 1, M + 1))
    for m in range(M + 1):
        for n in range(M + 1):
            if n > m:
                continue  # Upper triangular part remains zero
            sum_term = sum(
                (-1) ** j * comb(m, j) * ((1 - eta) + ((m - j) * eta / M)) ** n
                for j in range(m + 1)
            )
            C[m, n] = comb(M, m) * sum_term

    C_inv = np.linalg.inv(C)
    return C_inv


def stirling_first_kind(n, k):
    """Compute the signed Stirling number of the first kind using recursion."""
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    return (n - 1) * stirling_first_kind(n - 1, k) + stirling_first_kind(n - 1, k - 1)


def C_plus(k, m, N):
    """Compute C^+_{k,m} as defined in the given equation."""
    if k < 0 or k > N or m < 0 or m > k:
        return 0  # Out of valid range

    binomial_coeff = math.comb(N, k)
    stirling_number = stirling_first_kind(k, m)  # Compute signed Stirling number
    factor = (N**m) / math.factorial(k)

    return (1 / binomial_coeff) * factor * stirling_number


def n_choose_k(n, k):
    return math.comb(n, k)


def get_probs_from_clicks(click_probs, method="EME"):  # C_plus, EME
    """This follows arXiv:"""
    N = N_DET + 1
    click_vector = np.array(
        [
            click_probs[ph] if ph in list(click_probs.keys()) else ufloat(0, 0)
            for ph in np.arange(N)
        ]
    )

    if method == "C_plus":
        C = np.zeros((N, N))
        max_photons = len(click_probs.keys())
        for k in np.arange(N):
            for m in np.arange(N):
                C[k, m] = C_plus(k, m, N)

        print(C)
    elif method == "inversion_matrix":
        C = construct_C_matrix(N - 1, eta=1)
    elif method == "EME":
        n_max = 20
        det = DET(N_DET, n_max, eta=0.99)
        prob = EME(
            n_max, det=det, l=1e-3, c=np.array([x.nominal_value for x in click_vector])
        )
        return prob
    else:
        return np.array(
            [click_probs[ph] if ph in click_probs.keys() else 0 for ph in np.arange(N)]
        )

    prob = C @ click_vector

    return prob


################################################################
################################################################
################################################################


def measure_g2_vs_power(n_samples, tbi_params, power_array):

    for p in power_array:
        measure_g2(n_samples, tbi_params)


def measure_g2(n_samples, tbi_params):

    input_state = [1, 0]
    theta_list = [0]

    tbi = create_tbi(**tbi_params)
    start_time = time.time()
    samples = tbi.sample(
        input_state=input_state,
        theta_list=theta_list,
        n_samples=n_samples,
    )
    request_time = max([time.time() - start_time, 0.001])

    # calculate g(2)
    measure_heralded_g2(response, max_photons=5)
    measure_marginal_g2(response, max_photons=5)


def estimate_statistics(
    tbi, input_state, theta_list, n_samples, overhead_time_with_margin
):
    """This function runs an experiment on a particular tbi and
    returns the number of samples effectively collected, the rates, the accuracy,
    the time taken to collect the samples, and the samples themselves"""

    start_time = time.time()
    samples = tbi.sample(
        input_state=input_state,
        theta_list=theta_list,
        n_samples=n_samples,
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


def measure_heralded_g2(response, max_photons=5):

    # get coincidences
    samples = response.json()["results"]
    herald_data = response.json()["extra_data"]["n_heralds"]
    num_triggers = response.json()["extra_data"]["n_triggers"]

    if isinstance(samples, dict):
        c_h_s = {l: {} for l in samples.keys()}
        c_h = {}
        for k in samples.keys():
            c_h_s[k] = {
                ph: Counter(samples[k])[f"{ph}0"]
                for ph in np.arange(1, max_photons + 1)
            }
            c_h[k] = herald_data[k]
            c_h_s[k][0] = herald_data[k] - len(samples[k])

    else:
        c_h_s = {"A": {}}
        c_h = {}

        c_h_s["A"] = {
            ph: Counter(samples)[f"{ph}0"] for ph in np.arange(1, max_photons + 1)
        }
        c_h["A"] = herald_data["A"]
        c_h_s["A"][0] = herald_data["A"] - len(samples)

    g2_h = {}
    g2_h_alt = {}

    for k in c_h_s.keys():
        click_probs = {
            ph: ufloat(v, np.sqrt(v)) / ufloat(c_h[k], np.sqrt(c_h[k]))
            for ph, v in c_h_s[k].items()
        }
        probs = get_probs_from_clicks(click_probs, method="EME")
        n_av = np.sum([probs[n] * n for n in np.arange(1, len(probs))])
        g2_h[k] = (
            np.sum([probs[n] * n * (n - 1) for n in np.arange(2, len(probs))]) / n_av**2
        )

        ## ALTERNATIVE WAY ###
        num = ufloat(c_h[k], np.sqrt(c_h[k])) * ufloat(
            c_h_s[k][2], np.sqrt(c_h_s[k][2])
        )
        den = ufloat(c_h_s[k][1], np.sqrt(c_h_s[k][1]))

        g2_h_alt[k] = N_DET**2 / math.comb(N_DET, 2) * num / (den) ** 2

    print(g2_h, g2_h_alt)

    return g2_h


def measure_marginal_g2(response, max_photons=6):

    # get coincidences
    samples = response.json()["results"]
    num_triggers = response.json()["extra_data"]["n_triggers"]
    c_s = {}
    triggers_data = {}

    if isinstance(samples, dict):
        c_s1_s2 = {l: {} for l in samples.keys()}
        c_s = {}
        triggers_data = {}
        for k in samples.keys():
            c_s[k] = {
                ph: Counter(samples[k])[f"{ph}0"]
                for ph in np.arange(1, max_photons + 1)
            }
            triggers_data[k] = num_triggers[k]
            c_s[k][0] = triggers_data[k] - len(samples[k])

    else:
        c_s = {"A": {}}
        c_s["A"] = {
            ph: Counter(samples)[f"{ph}0"] for ph in np.arange(1, max_photons + 1)
        }
        triggers_data["A"] = num_triggers["A"]
        c_s["A"][0] = triggers_data["A"] - len(samples)

    g2_m = {}
    g2_m_alt = {}
    for k in c_s.keys():
        # reconstruct prob distribution
        click_probs = {
            ph: ufloat(v, np.sqrt(v))
            / ufloat(triggers_data[k], np.sqrt(triggers_data[k]))
            for ph, v in c_s[k].items()
        }
        probs = get_probs_from_clicks(click_probs, method="EME")
        n_av = np.sum([probs[n] * n for n in np.arange(1, len(probs))])
        print(probs, n_av)
        g2_m[k] = (
            np.sum([probs[n] * n * (n - 1) for n in np.arange(2, len(probs))]) / n_av**2
        )

        # the first factor comes from having multiple detectors
        num = ufloat(c_s[k][2], np.sqrt(c_s[k][2]))
        den = ufloat(c_s[k][1], np.sqrt(c_s[k][1]))
        g2_m_alt[k] = (
            N_DET**2 / math.comb(N_DET, 2) * num_triggers[k] * num / (den) ** 2
        )

    print("g2m")
    print(g2_m, g2_m_alt)
    return g2_m, n_av

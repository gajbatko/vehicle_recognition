import functools
import numpy as np
from scipy.signal import lfilter, firwin
from scipy.signal import medfilt


np.set_printoptions(threshold=np.nan)

def normalize_data(data):
    maximum = np.amax(data)
    minimum = np.amin(data)
    signal = [(x - minimum) / (maximum - minimum) for x in data]
    return signal

def data_cuter(data_r, data_x):
    data_r = data_r[300:12000]
    data_x = data_x[300:12000]

    #normalize signal r
    data_r_norm = normalize_data(data_r)

    #filter signal r
    fir_coeff = firwin(300, cutoff=0.01, window="hamming")
    filtered_signal = lfilter(fir_coeff, 1.0, data_r_norm)
    filtered_signal = filtered_signal[500:]
    filtered_signal = medfilt(filtered_signal, 301)
    filtered_signal = [i - filtered_signal[0] for i in filtered_signal]

    data_rev = filtered_signal[::-1]
    maximum = np.amax(filtered_signal)
    minimum = np.amin(filtered_signal)
    max_abs = max(maximum, minimum, key=abs)
    rev_average = functools.reduce(lambda x, y: x + y, filtered_signal[-100:]) / 100
    filter_treshold = max_abs * 15/100
    a = [n for n, i in enumerate(filtered_signal) if abs(i) > filter_treshold][0]
    b = len(filtered_signal) - [n for n, i in enumerate(data_rev) if abs(i-rev_average) > filter_treshold][0]

    difference_a = int(0.1 * (b - a))
    difference_b = int(0.1 * (b - a))


    if a - difference_a + 400 > 0:
        cut_data_r = data_r[a - difference_a+400:b + difference_b+401]
        cut_data_x = data_x[a - difference_a+400:b + difference_b+401]
    else:
        cut_data_r = data_r[a + 400:b + difference_b + 401]
        cut_data_x = data_x[a + 400:b + difference_b + 401]
    return cut_data_r, cut_data_x

def rmse(signal_1, signal_2, signal_3, signal_4):
    return np.sqrt(((signal_1 - signal_3) ** 2).mean()) + np.sqrt(((signal_2 - signal_4) ** 2).mean())

def avg_profile(dt_r_1, dt_r_2, dt_r_3):
    m_dt_r_1 = medfilt(dt_r_1, 51)
    m_dt_r_2 = medfilt(dt_r_2, 51)
    m_dt_r_3 = medfilt(dt_r_3, 51)

    averaged_profile_r = (m_dt_r_1 + m_dt_r_2 + m_dt_r_3) / 3

    return averaged_profile_r


def cut_with_median(dt_r_1, dt_x_1, dt_r_2, dt_x_2, dt_r_3, dt_x_3):
    dt_r_1, dt_x_1 = data_cuter(dt_r_1, dt_x_1)
    dt_r_2, dt_x_2 = data_cuter(dt_r_2, dt_x_2)
    dt_r_3, dt_x_3 = data_cuter(dt_r_3, dt_x_3)

    min_len = min([len(dt_r_1), len(dt_r_2), len(dt_r_3)])

    f = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
    c_dt_r_1 = [dt_r_1[i] for i in f(min_len, len(dt_r_1))]
    c_dt_r_2 = [dt_r_2[i] for i in f(min_len, len(dt_r_2))]
    c_dt_r_3 = [dt_r_3[i] for i in f(min_len, len(dt_r_3))]
    c_dt_x_1 = [dt_x_1[i] for i in f(min_len, len(dt_x_1))]
    c_dt_x_2 = [dt_x_2[i] for i in f(min_len, len(dt_x_2))]
    c_dt_x_3 = [dt_x_3[i] for i in f(min_len, len(dt_x_3))]

    c_dt_r_1 = normalize_data(c_dt_r_1)
    c_dt_x_1 = normalize_data(c_dt_x_1)
    m_dt_r_1 = medfilt(c_dt_r_1,51)
    m_dt_x_1 = medfilt(c_dt_x_1,51)

    c_dt_r_2 = normalize_data(c_dt_r_2)
    c_dt_x_2 = normalize_data(c_dt_x_2)
    m_dt_r_2 = medfilt(c_dt_r_2,51)
    m_dt_x_2 = medfilt(c_dt_x_2,51)

    c_dt_r_3 = normalize_data(c_dt_r_3)
    c_dt_x_3 = normalize_data(c_dt_x_3)
    m_dt_r_3 = medfilt(c_dt_r_3,51)
    m_dt_x_3 = medfilt(c_dt_x_3,51)

    averaged_profile_r = (m_dt_r_1 + m_dt_r_2 + m_dt_r_3) / 3
    averaged_profile_x = (m_dt_x_1 + m_dt_x_2 + m_dt_x_3) / 3

    sensors_signals = [[m_dt_r_1, m_dt_x_1], [m_dt_r_2, m_dt_x_2], [m_dt_r_3, m_dt_x_3]]

    rms_list_r = [rmse(x[0], x[1], averaged_profile_r, averaged_profile_x) for x in sensors_signals]

    network_sample_r = sensors_signals[rms_list_r.index(min(rms_list_r))][0]
    network_sample_x = sensors_signals[rms_list_r.index(min(rms_list_r))][1]

    return network_sample_r, network_sample_x

def network_input(dt_r_1, dt_x_1, dt_r_2, dt_x_2, dt_r_3, dt_x_3, filename):
    network_sample_r, network_sample_x = cut_with_median(dt_r_1, dt_x_1, dt_r_2, dt_x_2, dt_r_3, dt_x_3)

    f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]

    r_input_nodes = [network_sample_r[i] for i in f(50, len(network_sample_r))]
    x_input_nodes = [network_sample_x[i] for i in f(50, len(network_sample_x))]
    r_input_nodes.extend(x_input_nodes)

    output_file = open('passenger_cars_all.csv', 'a')
    input_data_str = ','.join(str(x) for x in r_input_nodes)
    input_data_str = ','.join([filename, input_data_str])
    output_file.write(','.join([input_data_str, 'passenger_car']) + "\n")

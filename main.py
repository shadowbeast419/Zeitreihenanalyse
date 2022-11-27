# This is a sample Python script.

import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz, filtfilt
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show


def plot_filter(w, h, nyq_rate, cutoff_hz, width, ripple_db):
    plt.plot((w / np.pi) * nyq_rate, 20 * np.log10(np.abs(h)), linewidth=2)

    plt.axvline(cutoff_hz + width * nyq_rate, linestyle='--', linewidth=1, color='g')
    plt.axvline(cutoff_hz - width * nyq_rate, linestyle='--', linewidth=1, color='g')
    plt.axhline(-ripple_db, linestyle='--', linewidth=1, color='c')
    delta = 10 ** (-ripple_db / 20)
    plt.axhline(20 * np.log10(1 + delta), linestyle='--', linewidth=1, color='r')
    plt.axhline(20 * np.log10(1 - delta), linestyle='--', linewidth=1, color='r')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Frequency Response')
    plt.ylim(-40, 5)
    plt.grid(True)


def filter_signal(x, is_low_pass, fs, cutoff_freq, plot):
    # FIR Filter
    nyq_rate = fs / 2.0

    # Width of the roll-off region.
    width = 1 / nyq_rate

    # Attenuation in the stop band.
    ripple_db = 40.0

    num_of_taps, beta = kaiserord(ripple_db, width)
    if num_of_taps % 2 == 0:
        num_of_taps = num_of_taps + 1

    cutoff_hz = 14
    taps = firwin(num_of_taps, cutoff_freq / nyq_rate, window=('kaiser', beta), pass_zero=is_low_pass)
    w, h = freqz(taps, worN=4000)

    if plot:
        plot_filter(w, h, nyq_rate, cutoff_hz, width, ripple_db)

    return filtfilt(taps, 1.0, x=x)

def calculate_fft(x, fft_samples, plot):
    # FFT Magic
    y_fft = np.fft.rfft(temperature_data, n=N)
    x_fft = np.fft.rfftfreq(N, d=1 / fs)

    if plot is True:
        plt.stem(x_fft, np.abs(y_fft) / n)
        plt.xlabel('f [1/Months]')
        plt.ylabel('FFT Amplitude |T(f)|')
        plt.title('FFT of T(t)')

    return x_fft, y_fft

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rich17dataset = xr.open_dataset('rich17.nc')

    # pressure_max = 85000 Pascal
    # latitude_max = 55.0
    # longitude_max = 35.0

    # pressure_min = 30000 Pascal
    # Index ist 7
    # Latitude / Longitude Gleich

    # sample frequency = 1 month
    fs = 32
    # FFT SAMPLE COUNT
    N = 128

    temperatures = rich17dataset.temperature.isel(pressure=11, longitude_bins=21, latitude_bins=14).to_numpy()
    time = rich17dataset.temperature.isel(pressure=11, longitude_bins=21, latitude_bins=14).time.to_numpy()

    frame = [temperatures, time, np.arange(0, time.size)]
    numpy_data = np.array(frame)

    df = pd.DataFrame(data=numpy_data).T
    df.columns = ['Temperature', 'Time', 'Months']

    rows_before_clear = df.shape[0]
    df = df.dropna()
    rows_after_clear = df.shape[0]

    offset = rows_before_clear - rows_after_clear

    plt.subplot(2, 2, 1)

    plt.plot(df['Months'] - offset, df['Temperature'])
    plt.title('Raw Temperature Data')
    plt.xlabel('t [Months]')
    plt.ylabel('T(t) [Kelvin]')

    # Remove Average
    temperature_data = df['Temperature'] - np.average(df['Temperature'])

    # Filter Signal (Aliasing Filter, Cutoff Frequency 14 Hertz)
    # temperature_data = filter_signal(temperature_data, True, fs, 14.0, True)
    # temperature_data = filter_signal(temperature_data, False, fs, 0.1, True)

    # Take every nth Element
    temperature_data = temperature_data[1::fs]
    months_data = df['Months'][1::fs] - offset

    # Signal length (amount of rows)
    n = temperature_data.shape[0]

    # Apply Hanning Window
    temperature_data = np.multiply(np.hanning(temperature_data.size), temperature_data)

    # Apply 1/n Normalization
    temperature_data = temperature_data / n

    plt.subplot(2, 2, 2)

    plt.plot(months_data, temperature_data)
    plt.title('Pre-Edited Temperature Data')
    plt.xlabel('t [Months]')
    plt.ylabel('T(t) [Kelvin]')

    # FFT Magic
    temperature_fft = np.fft.rfft(temperature_data, n=N)
    x_fft = np.fft.rfftfreq(N, d=1 / fs)

    # Shift the zero-frequency component to the center of the spectrum.
    # temperature_fft_shifted = np.fft.fftshift(temperature_fft)

    plt.subplot(2, 2, 3)

    plt.stem(x_fft, np.abs(temperature_fft) / n)
    plt.xlabel('f [1/Months]')
    plt.ylabel('FFT Amplitude |T(f)|')
    plt.title('FFT of T(t)')

    number_of_lags = 30
    temp_autocorr = sm.tsa.acf(df['Temperature'], nlags=number_of_lags)

    plt.subplot(2, 2, 4)

    plt.stem(np.arange(0, number_of_lags + 1, 1), temp_autocorr)
    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')
    plt.title('Autocorrelation of T(t)')

    plt.tight_layout()
    plt.show()

    figure()
    plt.subplot(2, 2, 1)
    temperature_data_filtered = filter_signal(temperature_data, is_low_pass=True, fs=fs, cutoff_freq=14, plot=True)
    plt.subplot(2, 2, 2)
    calculate_fft(temperature_data_filtered, fft_samples=N, plot=True)
    plt.subplot(2, 2, 3)
    temperature_data_filtered_filtered = filter_signal(temperature_data, is_low_pass=False, fs=fs, cutoff_freq=9, plot=True)
    plt.subplot(2, 2, 4)
    calculate_fft(temperature_data_filtered_filtered, fft_samples=N, plot=True)

    plt.tight_layout()
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

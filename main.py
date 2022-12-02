# This is a sample Python script.

import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz, filtfilt, butter
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_filter(data, cutoff, fs, order=5, is_lowpass = True, plot_enable = True):
    if is_lowpass is True:
        b, a = butter_lowpass(cutoff, fs, order=order)
    else:
        b, a = butter_highpass(cutoff, fs, order=order)

    if plot_enable is True:
        # Plotting the frequency response.
        w, h = freqz(b, a, worN=8000)
        plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5 * fs)
        plt.title("Filter Frequency Response, Cutoff_Freq = " + str(cutoff))
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.legend()
        plt.show()

    y = filtfilt(b, a, data)
    return y

def calculate_fft(x, fft_samples, sample_frequency, plot_enable, title_string):
    # FFT Magic
    y_fft = np.fft.rfft(x, n=fft_samples)
    x_fft = np.fft.rfftfreq(fft_samples, d=1.0 / sample_frequency)

    # Apply 1/n Normalization
    if plot_enable is True:
        plt.stem(x_fft, np.abs(y_fft) / x.shape[0])
        plt.xlabel('f [1/Months]')
        plt.ylabel('FFT Amplitude |T(f)|')
        plt.title(title_string)

    return x_fft, y_fft


def process_data_frame(data_frame, title_string):
    rows_before_clear = data_frame.shape[0]
    data_frame = data_frame.dropna()
    rows_after_clear = data_frame.shape[0]

    offset = rows_before_clear - rows_after_clear

    temperature_data = data_frame['Temperature']
    months_data = data_frame['Months'] - offset

    # Remove Average
    # temperature_data = data_frame['Temperature'] - np.average(data_frame['Temperature'])

    # Filter Signal (Aliasing Filter, Cutoff Frequency 14 Hertz)
    # temperature_data = filter_signal(temperature_data, True, fs, 14.0, True)
    # temperature_data = filter_signal(temperature_data, False, fs, 0.1, True)

    # Take every nth Element
    # temperature_data =
    # months_data = df['Months'][1::fs] - offset

    # Signal length (amount of rows)
    # n = temperature_data.shape[0]

    # sample frequency = 1 month
    fs = 1.0
    # FFT SAMPLE COUNT
    sample_count = 1024

    # Apply High pass to remove Offset
    data_without_bias = butter_filter(data=temperature_data, fs=fs,
                  cutoff=1.0 / 100.0, is_lowpass=False, plot_enable=True)

    plt.subplot(2, 1, 1)

    plt.plot(months_data, temperature_data)
    plt.title('Raw Temperature Data, ' + title_string)
    plt.xlabel('t [Months]')
    plt.ylabel('T(t) [Kelvin]')

    plt.subplot(2, 1, 2)

    # Apply Hanning Window
    data_without_bias_win = np.multiply(np.hanning(data_without_bias.size), data_without_bias)

    plt.plot(months_data, data_without_bias)
    plt.title('High Pass Temperature Data, ' + title_string)
    plt.xlabel('t [Months]')
    plt.ylabel('T(t) [Kelvin]')

    plt.tight_layout()
    plt.show()

    plt.plot(months_data, data_without_bias_win)
    plt.title('High Pass Temperature Data with Hanning Window, ' + title_string)
    plt.xlabel('t [Months]')
    plt.ylabel('T(t) [Kelvin]')
    plt.tight_layout()
    plt.show()

    plt.subplot(3, 1, 1)
    calculate_fft(x=temperature_data, sample_frequency=fs, fft_samples=sample_count,
                  plot_enable=True, title_string='T(f) FFT Unfiltered T(t), ' + title_string)

    plt.subplot(3, 1, 2)
    calculate_fft(x=data_without_bias, sample_frequency=fs, fft_samples=sample_count,
                  plot_enable=True, title_string='T(f) FFT of T(t) with Highpass, ' + title_string)

    plt.subplot(3, 1, 3)
    calculate_fft(x=data_without_bias_win, sample_frequency=fs, fft_samples=sample_count,
                  plot_enable=True, title_string='T(f) FFT of T(t) with Highpass, Hanning Window, ' + title_string)

    plt.tight_layout()
    plt.show()

    number_of_lags = 30
    temp_autocorr = sm.tsa.acf(temperature_data, nlags=number_of_lags)

    plt.stem(np.arange(0, number_of_lags + 1, 1), temp_autocorr)
    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')
    plt.title('Autocorrelation of raw T(t), ' + title_string)

    plt.tight_layout()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rich17dataset = xr.open_dataset('rich17.nc')

    # pressure_max = 85000 Pascal
    # latitude_max = 55.0
    # longitude_max = 35.0

    # pressure_min = 30000 Pascal
    # Index ist 7
    # Latitude / Longitude Gleich

    temperatures = rich17dataset.temperature.isel(pressure=11, longitude_bins=21, latitude_bins=14).to_numpy()
    time = rich17dataset.temperature.isel(pressure=11, longitude_bins=21, latitude_bins=14).time.to_numpy()

    frame = [temperatures, time, np.arange(0, time.size)]
    numpy_data = np.array(frame)

    df = pd.DataFrame(data=numpy_data).T
    df.columns = ['Temperature', 'Time', 'Months']

    process_data_frame(df, '85000 Pascal')

    temperatures2 = rich17dataset.temperature.isel(pressure=7, longitude_bins=21, latitude_bins=14).to_numpy()
    time2 = rich17dataset.temperature.isel(pressure=7, longitude_bins=21, latitude_bins=14).time.to_numpy()

    frame = [temperatures2, time2, np.arange(0, time2.size)]
    numpy_data = np.array(frame)

    df2 = pd.DataFrame(data=numpy_data).T
    df2.columns = ['Temperature', 'Time', 'Months']

    process_data_frame(df2, '30000 Pascal')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

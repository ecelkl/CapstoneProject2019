from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import csv
from matplotlib import pyplot as plt


def main2():
    fs_rate, signal = wavfile.read("/Users/elif//PyCharmProjects/Desktop/CapstoneProject/output.wav")

    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    print ("Timestep between samples Ts", Ts)

    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray

    AggressionLevel= (N / secs)
    print('Agg Level', AggressionLevel)

    if AggressionLevel<0:
      print("Aggression Exist!")
    elif AggressionLevel==0:
      print("Neutral")
    elif AggressionLevel > 0:
      print("Aggression Low")

    FFT = abs(scipy.fft(signal))
    FFT_side = FFT[range(N//2)] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)


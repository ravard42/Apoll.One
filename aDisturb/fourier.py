import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import scipy.io.wavfile


def fourier(sample):
	""" Compute the one-dimensional discrete Fourier Transform """
	mag = []
	fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
	for i in range(int(len(fft_data) / 2)):
		r = fft_data[i].real**2
		j = fft_data[i].imag**2
		mag.append(round(math.sqrt(r + j), 2))
	return mag

if __name__ == '__main__':
	if(len(sys.argv) != 2):
		print("usage: python fourier.py file.wav")
	else:
		rate, raw_audio = scipy.io.wavfile.read(sys.argv[1])
		raw_audio = [raw_audio[i][0] for i in range(len(raw_audio))]
		print("rate = ", rate)
		print("len(raw_audio) = ", len(raw_audio))
		mag = fourier(raw_audio)
		print("len(mag) = ", len(mag))
		plt.plot(mag)
		plt.show(mag)

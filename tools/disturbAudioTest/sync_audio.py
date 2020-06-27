"""
Code modified from https://github.com/allisonnicoledeal/VideoSync
"""
from multiprocessing.pool import ThreadPool
from subprocess import Popen, PIPE, call, check_output
import scipy.io.wavfile
import multiprocessing
import numpy as np
import datetime
import math
import json
import sys
import os
import matplotlib.pyplot as plt


threshold = 120
decal_slice_ratio = 1.75
duration = 4*60

def to_ffmpeg_format(time):
    h = time // 3600
    m = time // 60 - h * 60
    s = time - m * 60 - h * 3600
    return "%02d:%02d:%05.2f"%(h,m,s)

def cut_video(video, offset):
	str_cut_time = to_ffmpeg_format(offset)
	tmp_name = video + '.cut.mp4'
	print('CUTTING ' + video + ' at ' + str_cut_time + ' -> ' + tmp_name)   
	os.system("ffmpeg -y -ss " + str_cut_time + " -i " + video + " -c copy " + tmp_name + " -hide_banner")

def find_time_offset(video1, video2, left_shift, right_shift, audio_delays=(0., 0.), fft_bin_size=512, overlap=0, box_height=512, box_width=43, samples_per_box=7, durations=(duration, duration)):
	""" Find time offset between two video files and the frame rate. Assumes frame rate and audio bitrate of videos are the same.
	@param audio_delays: delay between video and audio, in audio_offset_s, in the first and second video files respectively
	@param fft_bin_size: size of the FFT bins, i.e. segments of audio, in beats, for which a separate peak is found
	@param overlap: overlap between each bin
	@param box_height: height of the boxes in frequency histograms
	@param box_width: width of the boxes in frequency histograms
	@param samples_per_box: number of frequency samples within each constellation
	@return time offset between the two videos in format (offset1, offset2)
	"""
	ft_dicts = []
	for vid, duration, shift in zip([video1, video2], durations, [left_shift, right_shift]):
		retcode, err_text, wavfile, _ = extract_audio(vid, duration, shift)
		if(retcode != 0):
			raise RuntimeError("ffmpeg error:\n{0:s}".format(err_text))
		rate, raw_audio = scipy.io.wavfile.read(wavfile)
		bins_dict = make_horiz_bins(raw_audio, fft_bin_size, overlap, box_height)  # bins, overlap, box height
		boxes = make_vert_bins(bins_dict, box_width)  # box width
		ft_dicts.append(find_bin_max(boxes, samples_per_box))  # samples per box
		if os.path.isfile(wavfile):
			os.remove(wavfile)  # Delete temporary wavefile

	# Determine time delay
	pairs = find_freq_pairs(*ft_dicts)
	delay, sync = find_delay(pairs)
	samples_per_sec = rate / (fft_bin_size - (overlap / 2))
	audio_offset_s = delay / samples_per_sec

	# Fix delays from manually measured delays between video and audio for both videos
	correction = audio_delays[1] - audio_delays[0]
	audio_offset_s = round(audio_offset_s, 4)

	if audio_offset_s > 0:
		return (0, audio_offset_s - correction, sync)
	else:
		return (abs(audio_offset_s) - correction, 0, sync)


def extract_audio(video_file, duration=-1, shift=0):
	""" Extract audio from video file and save it as wav audio file """
	audio_output_path = video_file[:video_file.index(".")] + ".wav"
	args = ["ffmpeg", "-y", "-ss", to_ffmpeg_format(shift), "-i", video_file, "-vn", "-ac", "1", "-f", "wav", "-hide_banner"]
	if duration > 0:
		duration_str = str(datetime.timedelta(seconds=duration)) + '.0'
		args.extend(["-t", duration_str])
	args.append(audio_output_path)
	process = Popen(args, stdout=PIPE, stderr=PIPE)
	output, err = process.communicate()
	exit_code = process.wait()
	return exit_code, str(err), audio_output_path, str(output)


def make_horiz_bins(data, fft_bin_size, overlap, box_height):
	horiz_bins = {}
	# process first sample and set matrix height
	sample_data = data[0:fft_bin_size]  # get data for first sample
	if (len(sample_data) == fft_bin_size):  # if there are enough audio points left to create a full fft bin
		intensities = fourier(sample_data)  # intensities is list of fft results
		for i in range(len(intensities)):
			box_y = int(i / box_height)
			if box_y in horiz_bins:
				horiz_bins[box_y].append((intensities[i], 0, i))  # (intensity, x, y)
			else:
				horiz_bins[box_y] = [(intensities[i], 0, i)]
	# process remainder of samples
	x_coord_counter = 1  # starting at second sample, with x index 1
	for j in range(int(fft_bin_size - overlap), len(data), int(fft_bin_size - overlap)):
		sample_data = data[j:j + fft_bin_size]
		if (len(sample_data) == fft_bin_size):
			intensities = fourier(sample_data)
			for k in range(len(intensities)):
				box_y = int(k / box_height)
				if box_y in horiz_bins:
					horiz_bins[box_y].append((intensities[k], x_coord_counter, k))  # (intensity, x, y)
				else:
					horiz_bins[box_y] = [(intensities[k], x_coord_counter, k)]
		x_coord_counter += 1
	return horiz_bins


def fourier(sample):
	""" Compute the one-dimensional discrete Fourier Transform """
	mag = []
	fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
	for i in range(int(len(fft_data) / 2)):
		r = fft_data[i].real**2
		j = fft_data[i].imag**2
		mag.append(round(math.sqrt(r + j), 2))
	return mag


def make_vert_bins(horiz_bins, box_width):
	boxes = {}
	for key in horiz_bins.keys():
		for i in range(len(horiz_bins[key])):
			box_x = int(horiz_bins[key][i][1] / box_width)
			if (box_x, key) in boxes:
				boxes[(box_x, key)].append((horiz_bins[key][i]))
			else:
				boxes[(box_x, key)] = [(horiz_bins[key][i])]
	return boxes


def find_bin_max(boxes, maxes_per_box):
	"""
	On rÃ©cupÃ¨re l'histogramme (t,f) -> [(i, t, f)]
	On crÃ©e un dictionnaire des frÃ©quences qui domminent
	(le groupe des maxes_per_box plus intenses)
	en intensitÃ© leur tranche temporelle
	On obtient un dictionnaire f -> t
	"""
	freqs_dict = {}
	for key in boxes.keys():
		max_intensities = [(1, 2, 3)]
		for i in range(len(boxes[key])):
			if boxes[key][i][0] > min(max_intensities)[0]:
				if len(max_intensities) < maxes_per_box:  # add if < number of points per box
					max_intensities.append(boxes[key][i])
				else:  # else add new number and remove min
					max_intensities.append(boxes[key][i])
					max_intensities.remove(min(max_intensities))
		for j in range(len(max_intensities)):
			if max_intensities[j][2] in freqs_dict:
				freqs_dict[max_intensities[j][2]].append(max_intensities[j][1])
			else:
				freqs_dict[max_intensities[j][2]] = [max_intensities[j][1]]
	return freqs_dict


def find_freq_pairs(freqs_dict_orig, freqs_dict_sample):
	time_pairs = []
	for key in freqs_dict_sample.keys():  # iterate through freqs in sample
		if key in freqs_dict_orig:  # if same sample occurs in base
			for i in range(len(freqs_dict_sample[key])):  # determine time offset
				for j in range(len(freqs_dict_orig[key])):
					time_pairs.append((freqs_dict_sample[key][i], freqs_dict_orig[key][j]))
	return time_pairs


def find_delay(time_pairs):
	t_diffs = {}
	for i in range(len(time_pairs)):
		delta_t = time_pairs[i][0] - time_pairs[i][1]
		if delta_t in t_diffs:
			t_diffs[delta_t] += 1
		else:
			t_diffs[delta_t] = 1

	t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1], reverse=True)
	print(t_diffs_sorted[0:20])
	time_delay = t_diffs_sorted[0][0]
	
	#sync = max(t_diffs.values()) > threshold and t_diffs_sorted[-99][1] < 1 / 3 * t_diffs_sorted[-1][1]


	a = [i[0] for i in t_diffs_sorted[:5]]
	cond_a = np.std(a) < 30.0
	col_a = '\033[92m' if cond_a else '\033[91m' 
	b = [i[1] for i in t_diffs_sorted[:5]]
	cond_b = max(b) - min(b) > max(b) / 5
	col_b = '\033[92m' if cond_b else '\033[91m' 
	sync = cond_a and cond_b
	col = '\033[92m' if sync else '\033[91m' 
	print("a:", a)
	print("standard deviation =" + col_a, np.std(a), '\033[0m')
	print("b", b)
	print("max(b) / 5=" + col_b, max(b) / 5, "\033[m")
	print("max(b) - min(b) =" + col_b, max(b) - min(b), "\033[m")
	print("recup: " + col, sync, '\033[0m')

	plt.scatter([i[0] for i in t_diffs_sorted[20:]], [i[1] for i in t_diffs_sorted[20:]], c='b', edgecolors='black')
	plt.scatter([i[0] for i in t_diffs_sorted[5:20]], [i[1] for i in t_diffs_sorted[5:20]], c='r', edgecolors='black')
	plt.scatter([i[0] for i in t_diffs_sorted[:5]], [i[1] for i in t_diffs_sorted[:5]], c='g', edgecolors='black')
	plt.show()
	return time_delay, sync

def   is_valid_slice(video, slice_time):
	video_duration = float(check_output(['ffprobe', '-v',  'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video]))
	if (slice_time - video_duration >= 0.05):
		return False
	return True

def	sync_audio(lvid, rvid):
	sync = False
	for s in range(20):
		for n in range(s+1):
			l = s-n
			r = n
			print("shift : %d - %d" % (l, r))
			lt = l * decal_slice_ratio  * duration
			rt = r * decal_slice_ratio  * duration
			if (not is_valid_slice(lvid, lt) or not is_valid_slice(rvid, rt)):
				continue
			offset1, offset2, sync = find_time_offset(lvid, rvid, lt, rt, durations=(duration, duration))
			print("check", sync)
			offset1 += lt
			offset2 += rt
			if sync:
				break
		if sync:
			break
	if (sync):
		offset = abs(offset1 - offset2)
		if (offset1 >= offset2):
			cut_video( lvid, offset)
			cut_video( rvid, 0)
		else:
			cut_video( rvid, offset)
			cut_video( lvid, 0)
	else:
		print("sync failed")
	return offset1, offset2, sync

if __name__ == '__main__':
	if(len(sys.argv) != 3):
		print('You should provide exactly two arguments (videos path).')
	else:
		sync_audio(sys.argv[1], sys.argv[2])

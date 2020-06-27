"""
Code modified from https://github.com/allisonnicoledeal/VideoSync
"""
from multiprocessing.pool import ThreadPool
from subprocess import Popen, PIPE, call
import scipy.io.wavfile
import multiprocessing
import numpy as np
import datetime
import math
import json
import sys
import os

#import utils

__all__ = ['cut_video', 'find_time_offset']

def to_ffmpeg_format(time):
    h = time // 3600
    m = time // 60 - h * 60
    s = time - m * 60 - h * 3600
    return "%02d:%02d:%05.2f"%(h,m,s)

def cut_video(video, offset):
    tmp_name = video + '.cut.mp4'
    #str_offset = str(datetime.datetime.fromtimestamp(offset).time())
    str_offset = "00:00:"+str(offset)
    call(['ffmpeg', '-i', video, '-ss', str_offset, '-async', '1', '-c', 'copy', tmp_name])
    #os.remove(video)
    #os.rename(tmp_name, video)


def find_time_offset(video1, video2, audio_delays=(0., 0.), fft_bin_size=512, overlap=0, box_height=512, box_width=43, samples_per_box=7, durations=(120, 120)):
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
    for vid, duration in zip([video1, video2], durations):
        retcode, err_text, wavfile, _ = extract_audio(vid, duration)
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
    delay = find_delay(pairs)
    samples_per_sec = rate / (fft_bin_size - (overlap / 2))
    audio_offset_s = delay / samples_per_sec

    # Fix delays from manually measured delays between video and audio for both videos
    correction = audio_delays[1] - audio_delays[0]
    audio_offset_s = round(audio_offset_s, 4)

    if audio_offset_s > 0:
        return (0, audio_offset_s - correction)
    else:
        return (abs(audio_offset_s) - correction, 0)


def extract_audio(video_file, duration=-1):
    """ Extract audio from video file and save it as wav audio file """
    audio_output_path = video_file[:video_file.index(".")] + ".wav"
    args = ["ffmpeg", "-y", "-i", video_file, "-vn", "-ac", "1", "-f", "wav"]
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
    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
    time_delay = t_diffs_sorted[-1][0]
    return time_delay


if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('You should provide exactly two arguments (videos path).')
    else:
        offset1, offset2 = find_time_offset(sys.argv[1], sys.argv[2])
        print('Offset1: ' + to_ffmpeg_format(offset1))
        print('Offset2: ' + to_ffmpeg_format(offset2))
        #cut_video(sys.argv[1], offset1)
        #cut_video(sys.argv[2], offset2)

"""
Code modified from https://github.com/allisonnicoledeal/VideoSync
"""
from multiprocessing.pool import ThreadPool
from subprocess import Popen, PIPE, call, STDOUT, check_output
import scipy.io.wavfile
import multiprocessing
import numpy as np
import datetime
import math
import json
import sys
import os
import matplotlib.pyplot as plt
import re

duration = 60

__all__ = ['to_ffmpeg_format', 'find_time_offset']

from sync_audio import to_ffmpeg_format, find_time_offset

def detect_disturb(video1, video2):
	time1 = float(check_output(['ffprobe', '-v',  'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', sys.argv[1]]))
	time2 = float(check_output(['ffprobe', '-v',  'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', sys.argv[2]]))
	time =  min(time1 - 60, time2 - 60)

	print("\nworking at :" + '\033[94m', to_ffmpeg_format(time), "\33[0m")
	offset1, offset2, sync = find_time_offset(
		video1, video2, 
		time, time, 
		durations=(duration, duration))
	if sync and abs(offset1 - offset2) < 1:
		print("Pas de perturbations dans la vidéo")
		return
	bloc_size = time // 2 # Taille du bloc où se trouve la perturbation
	time = time // 2

	for i in range(10):
		print("\nworking at :" + '\033[94m', to_ffmpeg_format(time), "\33[0m")
		offset1, offset2, sync = find_time_offset(
			video1, video2, 
			time, time, 
			durations=(duration, duration))
		if sync and abs(offset1 - offset2) < 1:
			# La perturbation est après (entre time et time + bloc_size)
			bloc_size = bloc_size // 2 # La taille du bloc de recherche est divisée par deux
			time += bloc_size # On se place au milieu du bloc d'après
		else:
			bloc_size = bloc_size // 2 # La taille du bloc de recherche est divisée par deux
			time -= bloc_size # On se place au milieu du bloc d'avant
	
	print("detectAudioDisturb returned:", to_ffmpeg_format(time))
	return to_ffmpeg_format(time)

if __name__ == '__main__':
	if(len(sys.argv) != 3):
		print('You should provide exactly two arguments (videos path).')
	else:
		detect_disturb(sys.argv[1], sys.argv[2])

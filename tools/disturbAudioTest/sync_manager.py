from sync_audio import sync_audio
from detectAudioDisturb import detect_disturb
from multiprocessing.pool import ThreadPool
from subprocess import Popen, PIPE, call
import sys
import os

if __name__ == '__main__':
	video1 = sys.argv[1]
	video2 = sys.argv[2]
	
	sync_audio(video1, video2)
	perturb_time = detect_disturb(video1+".cut.mp4", video2+".cut.mp4")
	print(perturb_time)
	if perturb_time is not None:
		call(['ffmpeg', '-t', perturb_time, '-i', video1+".cut.mp4", '-async', '1', '-c', 'copy', video1+".0.mp4", "-hide_banner"])
		call(['ffmpeg', '-t', perturb_time, '-i', video2+".cut.mp4", '-async', '1', '-c', 'copy', video2+".0.mp4", "-hide_banner"])
		call(['ffmpeg', '-ss', perturb_time, '-i', video1+".cut.mp4", '-async', '1', '-c', 'copy', video1+".1.mp4", "-hide_banner"])
		call(['ffmpeg', '-ss', perturb_time, '-i', video2+".cut.mp4", '-async', '1', '-c', 'copy', video2+".1.mp4", "-hide_banner"])
		sync_audio(video1+".1.mp4", video2+".1.mp4")
		with open("concat_left.txt", "w") as f:
			f.write("file '%s' \n"%(video1+".0.mp4"))
			f.write("file '%s' \n"%(video1+".1.mp4.cut.mp4"))
		with open("concat_right.txt", "w") as f:
			f.write("file '%s' \n"%(video2+".0.mp4"))
			f.write("file '%s' \n"%(video2+".1.mp4.cut.mp4"))
		call(['ffmpeg', '-y', '-f', 'concat', '-i', "concat_left.txt", '-c', 'copy', video1+".fin.mp4", "-hide_banner"])
		call(['ffmpeg', '-y', '-f', 'concat', '-i', "concat_right.txt", '-c', 'copy', video2+".fin.mp4", "-hide_banner"])
		


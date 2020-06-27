nohup ffmpeg -f concat -safe 0 -i concat.txt -c copy $1.mp4 > $1.log 2>&1 &

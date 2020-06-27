nohup ffmpeg -f concat -safe 0 -i concat_left.txt -c copy left.mp4 > left.log 2>&1 &
nohup ffmpeg -f concat -safe 0 -i concat_right.txt -c copy right.mp4 > right.log 2>&1 &
#nohup ffmpeg -f concat -safe 0 -i concat.txt -c copy $1.mp4 > $1.log 2>&1 &

#ffmpeg -f concat -safe 0 -i concat.txt -c copy $1
ffmpeg -f concat -safe 0 -i concat_left.txt -c copy left.mp4 &&
ffmpeg -f concat -safe 0 -i concat_right.txt -c copy right.mp4

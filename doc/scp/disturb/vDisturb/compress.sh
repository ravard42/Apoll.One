nohup ffmpeg -i $1/left.mp4 -vf "scale=iw/4:ih/4" $1/compressLeft.mp4 > $1/compressLeft.log 2>&1 &
nohup ffmpeg -i $1/right.mp4 -vf "scale=iw/4:ih/4" $1/compressRight.mp4 > $1/compressRight.log 2>&1 &

if [ $# != 1 ]
then
echo "usage : sh cut.sh fileName.mp4"

else
TMP=output.mp4

ffmpeg -i $1 -vf "hflip,vflip,format=yuv420p" -metadata:s:v rotate=0 -codec:v libx264 -codec:a copy $TMP
mv $TMP $1


fi

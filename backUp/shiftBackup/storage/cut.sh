if [ $# != 1 ]
then
echo "usage : sh cut.sh fileName.mp4"

else
TMP=output.mp4

ffmpeg  -i $1 -ss 00:00:06 -to 00:30:00 -c copy $TMP
mv $TMP $1


fi

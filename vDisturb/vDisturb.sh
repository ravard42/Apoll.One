if [ $# != 1 ] 
then
echo "usage: sh vDisturb.sh MATCH_DIR_PATH (with left.mp4 et right.mp4 in MATCH_DIR_PATH)"
else

#WITH COMPRESSION

nohup ffmpeg -y -i $1/left.mp4.cut.mp4 -vf "scale=iw/4:ih/4" $1/compressLeft.mp4 > $1/compressLeft.log 2>&1 &
nohup ffmpeg -y -i $1/right.mp4.cut.mp4 -vf "scale=iw/4:ih/4" $1/compressRight.mp4 > $1/compressRight.log 2>&1 &
wait

#nohup ./detectVideoDisturb $1/left.mp4 > /dev/null 2>&1 &
#nohup ./detectVideoDisturb $1/right.mp4 > /dev/null 2>&1 &
nohup ./detectVideoDisturb $1/compressLeft.mp4 > /dev/null 2>&1 &
nohup ./detectVideoDisturb $1/compressRight.mp4 > /dev/null 2>&1 &
#nohup ./detectVideoDisturb $1/compressLeft.mp4 > detectLeft.log 2>&1 &
#nohup ./detectVideoDisturb $1/compressRight.mp4 > detectRight.log 2>&1 &
wait

nohup ./fusion_v_log $1 > /dev/null 2>&1
rm $1/compressLeft.vDist.log $1/compressRight.vDist.log

echo "vDisturb.sh just finished see $1/vDist.log"




#W/O COMPRESSION
#
#nohup ./detectVideoDisturb $1/left.mp4 > /dev/null 2>&1 &
#nohup ./detectVideoDisturb $1/right.mp4 > /dev/null 2>&1 &&
#nohup ./fusion_v_log $1 > /dev/null 2>&1
#
#rm $1/left.vDist.log $1/right.vDist.log
#
#echo "vDisturb.sh w/o compression just finished see MATCH_DIR_PATH/vDist.log"
fi

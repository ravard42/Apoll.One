sync_audio='/home/apollone/ravard/shift/storage/aDisturb/sync_audio.py'
detectAudioDisturb='/home/apollone/ravard/shift/storage/aDisturb/detectAudioDisturb.py'

if [ $# != 1 ] && [ $# != 2 ];then
echo "usage: sh aSync.sh MATCH_DIR_PATH (with left.mp4 et right.mp4 in MATCH_DIR_PATH) title (optional)"
else

title="default"

if [ $# = 2 ];then
title=$2
fi


nohup python3 $sync_audio $1/left.mp4 $1/right.mp4 $title < /dev/null > $1/aSync.log 2> $1/errASync.log  &&
nohup python3 $detectAudioDisturb $1/left.mp4.cut.mp4 $1/right.mp4.cut.mp4 $title < /dev/null >> $1/aSync.log 2>> $1/errASync.log  &&

echo "aSync.sh just finished see MATCH_DIR_PATH/aSync.log and MATCH_DIR_PATH/errASync.log"
fi

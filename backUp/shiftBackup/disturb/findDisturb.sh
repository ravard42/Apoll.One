if [ $# != 1 ] 
then
echo "usage: sh findDisturb.sh MATCH_DIR_PATH (with left.mp4 et right.mp4 in MATCH_DIR_PATH)"
else
nohup ./findDisturb $1/left.mp4 > /dev/null 2>&1 &
nohup ./findDisturb $1/right.mp4 > /dev/null 2>&1 &&
nohup ./logFusion $1 > /dev/null 2>&1

rm $1/left.disturb.log $1/right.disturb.log

echo "findDisturb just finished see MATCH_DIR_PATH/fusion.disturb.log"
fi

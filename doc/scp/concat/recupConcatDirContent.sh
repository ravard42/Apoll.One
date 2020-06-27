SRCPATH=/mnt/disks/videos/ravard/concat
EXEC=recupConcatDirContent.sh

if [ $# = 1 ]
then
gcloud compute scp --recurse $1:$SRCPATH ./
else
echo "usage : sh $EXEC [cvNumber]"
fi

SRCPATH=/home/micka/trackingvideo/stitching/ravard/verifPano
EXEC=recupVerifPanoDirContent.sh

if [ $# = 1 ]
then
gcloud compute scp --recurse $1:$SRCPATH/* ./
elif [ $# = 2 ]
then
gcloud compute scp --recurse $1:$SRCPATH/$2 ./
else
echo "usage : sh $EXEC [cvNumber]"
echo "usage : sh $EXEC [cvNumber] [directory name]"
fi

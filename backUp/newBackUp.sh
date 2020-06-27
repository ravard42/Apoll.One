mkdir -p shiftBackup
mkdir -p stitchBackup

gcloud compute scp --recurse $1:/mnt/disks/videos/ravard/* ./shiftBackup
gcloud compute scp --recurse $1:/home/micka/trackingvideo/stitching/ravard/* ./stitchBackUp

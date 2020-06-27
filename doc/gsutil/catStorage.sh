gsutil ls -lah gs://clients_apollone/* | grep  $1
gsutil ls -lah gs://clients_apollone/* | grep -c $1

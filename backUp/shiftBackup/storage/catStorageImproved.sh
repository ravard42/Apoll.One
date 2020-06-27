gsutil ls -lah gs://$1/* | grep  $2
gsutil ls -la gs://$1/* | grep -c $2

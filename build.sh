#!/bin/bash

echo "Choice to delete rest-server tag in Local Docker Repository (tag name / n)"
curl http://175.193.37.235:8942/v2/lotto-ml/tags/list

# shellcheck disable=SC2162
read deleteTag

case $deleteTag in
n | No | N)
    echo "Skip Delete tag in Local Docker Repository..."
    ;;
*)
    echo "Delete tag in Local Docker Repository..."
    DIGEST=$(curl -v --silent -H "Accept: application/vnd.docker.distribution.manifest.v2+json" -X GET http://175.193.37.235:8942/v2/lotto-ml/manifests/"$deleteTag" 2>&1 | grep Docker-Content-Digest | awk '{print $3}' | tr -d '\n' | tr -d '\r')
    curl -X DELETE "http://175.193.37.235:8942/v2/lotto-ml/manifests/$DIGEST"
#     ssh yousang@175.193.37.235 -p 3258 "eval /usr/local/bin/minikube docker-env && docker exec -it my-registry bin/registry garbage-collect /etc/docker/registry/config.yml"
    ;;
esac

echo "Please enter the image tag:"
# shellcheck disable=SC2162
read tag

echo "Pushing to Local Docker Repository..."
docker build --no-cache --platform linux/amd64 -t 175.193.37.235:8942/lotto-ml:"$tag" .
docker push 175.193.37.235:8942/lotto-ml:"$tag"
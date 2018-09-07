#!/bin/bash

PROJECT="formula-trend"
REGISTRY="ai.registry.trendmicro.com"

# Verify docker installation
which docker > /dev/null
RETVAL=$?
if [ $RETVAL -ne 0 ] ; then
    echo "Docker is not installed yet, please install it first." 
    echo "Reference:" 
    echo "  https://wiki.jarvis.trendmicro.com/display/2SC/How+to+Build+a+Docker+Image#HowtoBuildaDockerImage-InstallDocker"
    exit 1 
fi

# Identify your Project Name and login to ai.registry
echo -n "Please enter your team number "
echo -n "(Refer to Your Project Name on https://$REGISTRY/): "
read project_name
echo -n "Your team number is '$project_name', is it correct? (yes or no) "
read answer
if [[ $answer =~ [Yy][Ee][Ss] ]] ; then
    echo "Please enter your username and password to login $REGISTRY"
    docker login $REGISTRY || (echo "Exit due to login docker hub failed." && exit 1)
else 
    echo "Exit due to incorrect team number." 
    exit 1
fi 

practice_tag="$REGISTRY/$project_name/$PROJECT:practice"
rank_tag="$REGISTRY/$project_name/$PROJECT:rank"

docker build -t $PROJECT .
for tag in $practice_tag $rank_tag;
do
    docker tag $PROJECT $tag
    docker push $tag
    docker rmi $tag
done

echo "Clean up your env"
# tags=$(docker history $(docker images -a | grep 'practice'  | awk '{print $3}') | awk '{print $1}' | grep -v "missing\|IMAGE")
# docker rmi -f $tags
docker rmi $PROJECT
docker logout $REGISTRY

echo ""
echo "Upload complete"
echo "Run \"docker login $REGISTRY; docker run -ti -p 4567:4567 $rank_tag\" to do local testing"

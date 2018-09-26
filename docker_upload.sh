#!/bin/bash

PROJECT="formula-trend"
REGISTRY="ai.registry.trendmicro.com"
DOCKER_TEMPLATE="Dockerfile.template"
DOCKERFILE="Dockerfile"
# Verify docker installation
which docker > /dev/null
RETVAL=$?
if [ $RETVAL -ne 0 ] ; then
    echo "Docker is not installed yet, please install it first." 
    echo "Reference:" 
    echo "  https://wiki.jarvis.trendmicro.com/display/2SC/How+to+Build+a+Docker+Image#HowtoBuildaDockerImage-InstallDocker"
    exit 1 
fi
# create a docker file from template
# we only add ENTRYPOINT at this time
# Identify which bot you want to use
echo -n "Please choose the bot you want to use? (1) bc, (2) pid, (3) mpc "
read bot_type
re='^[0-9]+$'
if ! [[ $bot_type =~ $re ]] ; then
   echo "error: Not a number: " $bot_type; exit 1
fi
if [ $bot_type -lt 1 ] || [ $bot_type -gt 3 ] ;  then
    echo "Exit due to incorrect bot type."
    exit 1
else
    echo "you choose" $bot_type
    if [ -f $DOCKERFILE ]; then
        echo "rm" $DOCKERFILE
        rm $DOCKERFILE
    fi
    
    echo "cp " $DOCKER_TEMPLATE $DOCKERFILE
    cp $DOCKER_TEMPLATE $DOCKERFILE
    if [ $bot_type == 1 ] ; then
        echo "ENTRYPOINT [\"python3\", \"bot.py\", \"-b\", \"bc\"]" >> $DOCKERFILE
    elif [ $bot_type == 2 ] ; then
        echo "ENTRYPOINT [\"python3\", \"bot.py\", \"-b\", \"pid\"]" >> $DOCKERFILE
    else
        echo "ENTRYPOINT [\"python3\", \"bot.py\", \"-b\", \"mpc\"]" >> $DOCKERFILE
    fi
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
for tag in $practice_tag $rank_tag ;
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
if [ -f $DOCKERFILE ]; then
    echo "rm" $DOCKERFILE
    rm $DOCKERFILE
fi
echo ""
echo "Upload complete"
echo "Run \"docker login $REGISTRY; docker run -ti -p 4567:4567 $rank_tag\" to do local testing"

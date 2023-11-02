docker run --user "$(id -u)":"$(id -g)" --runtime=nvidia --gpus all -d --mount type=bind,src=/"$(pwd)/",target=/a2 -it --name $1 nlp-a2-2324 python3 runnables/$2

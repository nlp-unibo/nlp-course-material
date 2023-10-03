docker run --user "$(id -u)":"$(id -g)" --runtime=nvidia --gpus all -d --mount type=bind,src=/"$(pwd)/",target=/sp -it --name $1 nlp-sp-2324 python3 runnables/$2

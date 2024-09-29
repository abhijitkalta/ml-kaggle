#Now, we will create a docker file called Dockerfile. No extension. There are several
#elements to Dockerfile. Let’s take a look.
# Dockerfile
# First of all, we include where we are getting the image
# from. Image can be thought of as an operating system.
# You can do "FROM ubuntu:18.04"
# this will start from a clean ubuntu 18.04 image.
# All images are downloaded from dockerhub
# Here are we grabbing image from nvidia's repo
# they created a docker image using ubuntu 18.04
# and installed cuda 10.1 and cudnn7 in it. Thus, we don't have to
# install it. Makes our life easy.
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# this is the same apt-get command that you are used to
# except the fact that, we have -y argument. Its because
# when we build this container, we cannot press Y when asked for
RUN apt-get update && apt-get install -y \
git \
curl \
ca-certificates \
python3 \
sudo \
&& rm -rf /var/lib/apt/lists/*
# We add a new user called "abhijit"
# this can be anything. Anything you want it
# to be. Usually, we don't use our own name,
# you can use "user" or "ubuntu"
RUN useradd -m abhijit
# make our user own its own home directory
RUN chown -R abhijit:abhijit /home/abhijit/
# copy all files from this direrctory to a
# directory called app inside the home of abhijit
# and abhijit owns it.
COPY --chown=abhijit *.* /home/abhijit/app/
# change to user abhijit
USER abhijit
RUN mkdir /home/abhijit/data/
# Now we install all the requirements
# after moving to the app directory
# PLEASE NOTE that ubuntu 18.04 image
# has python 3.6.9 and not python 3.7.6
# you can also install conda python here and use that
# however, to simplify it, I will be using python 3.6.9
# inside the docker container!!!!
RUN cd /home/abhijit/app/ && pip3 install -r requirements.txt
# install mkl. its needed for transformers
RUN pip3 install mkl
# when we log into the docker container,
# we will go inside this directory automatically
WORKDIR /home/abhijit/app

# Once we have created the docker file, we need to build it. Building the docker

# container is a very simple command.
# docker build -f Dockerfile -t bert:train .
# Please note that I have removed many lines from the output. Now, you can log into
# the container using the following command.
# $ docker run -ti bert:train /bin/bash
# You need to remember that whatever you do in this shell will be lost once you exit
# the shell. And you can run the training inside the docker container using:
# $ docker run -ti bert:train python3 train.py
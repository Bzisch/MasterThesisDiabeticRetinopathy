#!/bin/bash -e

	mkdir -p data/training_0
	mkdir -p data/training_0/train_sm
	mkdir -p data/training_0/train_sm_180
	mkdir -p data/training_0/train_sm_flop
	mkdir -p data/training_0/train_sm_180_flop


	convert $1 -fuzz 10% -trim -resize 256x256 \
		      -background black -gravity center -extent 256x256 \
		          data/training_0/train_sm/$(basename -s .jpeg $1).png

	convert $1 -fuzz 10% -trim -resize 256x256 \
		       -background black -gravity center -extent 256x256 \
		-rotate 180 data/training_0/train_sm_180/$(basename -s .jpeg $1).png

	convert $1 -fuzz 10% -trim -resize 256x256 \
		       -background black -gravity center -extent 256x256 \
		-flop data/training_0/train_sm_flop/$(basename -s .jpeg $1).png

	convert $1 -fuzz 10% -trim -resize 256x256 \
	               -background black -gravity center -extent 256x256 \
	       	-rotate 180 -flop data/training_0/train_sm_180_flop/$(basename -s .jpeg $1).png
	echo $1 

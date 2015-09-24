#!bin/bash

convert $1 -rotate 180 resized/train_sm_180/$(basename -s .jpeg $1).png
convert $1  -flop resized/train_sm_flop/$(basename -s .jpeg $1).png
convert $1 -rotate 180 -flop resized/train_sm_180_flop/$(basename -s .jpeg $1).png


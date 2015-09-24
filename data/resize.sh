#!/bin/bash -e
# ls train/*.jpeg | parallel sh resize.sh

size=256x256

out_dir=resized/
mkdir -p $out_dir/trainOriginal256
out=$out_dir/$1
[ -e $out ] && echo "Skip $1" || echo "$1 -> $out"
[ -e $out ] || \
convert -fuzz 10% -trim +repage -resize $size -gravity center -background black -extent $size $1 $out




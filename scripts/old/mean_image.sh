#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=../../data/osnovna_baza
TOOLS=../caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE \
	$EXAMPLE/mean.prototxt
echo "Done."

#!bin/bash -e

# skip first line, adjust paths, add extension, treat label {1,..,4} as 1
tail -n +2 trainLabels.csv | \ 
          sed 's/,/.png\t/g' | \ 
            sed 's#^#/home/jbzik/Documents/Diplomski_Bzik/jbzik_kaggle_data/data/data/training_0/train_sm/#g' | \
              sed 's/[234]$/1/g' \
                > imgs.csv
                    
# manual intermediate step:
# remove images which caused errors during above transformations (about 15)

# create four lists (img,label) for all four versions
cp imgs.csv train_sm.csv
sed 's/train_sm/train_sm_180/g' imgs.csv > train_sm_180.csv
sed 's/train_sm/train_sm_flop/g' imgs.csv > train_sm_flop.csv
sed 's/train_sm/train_sm_180_flop/g' imgs.csv > train_sm_180_flop.csv

# combine, shuffle and save in one file
cat train_sm*.csv | shuf > train_all.csv

# first 2000 images for testing
# next 6000 images for validation
# rest for training
sed -n '1,2000p' <  train_all.csv > testing.csv
sed -n '2001,8000p' <  train_all.csv > validation.csv
sed -n '8001,$p' <  train_all.csv > training.csv


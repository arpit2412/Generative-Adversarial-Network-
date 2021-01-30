#!/bin/bash
#!/bin/bash
#!/bin/bash
#SBATCH -p batch # partition (this is the queue your job will be added to)
#SBATCH -n 2 # number of cores (here 2 cores requested)
#SBATCH --time=02:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:2 # generic resource required (here requires 1 GPU)
#SBATCH --mem=32GB # specify memory required per node (here set to 16 GB)

# Notification configuration
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL) 
#SBATCH --mail-user=arpit.garg@adelaide.edu.au          # Email to which notifications will be sent 
# Execute your script
FILE=$1

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./$FILE.zip
TARGET_DIR=./$FILE
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d .
rm $ZIP_FILE

# Adapt to project expected directory heriarchy
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"

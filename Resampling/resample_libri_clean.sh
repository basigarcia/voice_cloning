#!/bin/bash

STARTFOLDER="LibriSpeech"
NEWFOLDER="LibriSpeech_8k"
PATTERN="*"
FORMAT="flac"

# Iterate the string variable using for loop
# for val in $StringVal; do
#     echo $val
# done

echo "Finding candidate files .."
FILELIST=$(find "$STARTFOLDER/" -type f -name "*.$FORMAT")
echo ".. "$(echo -e "$FILELIST" | wc -l)" found"
mkdir -p $NEWFOLDER

for file in $FILELIST; do
  # A bit of healthy paranoia
  test -z "$file" && continue
  test -f "$file" || continue
  # echo "Working on $file .."
  
  # New file name
  BN=$(basename "$file" ".$FORMAT")
  BP="$NEWFOLDER/$(dirname "$file")"
  mkdir -p "$BP"
  NEWFILEPATH="$BP/$BN.$FORMAT"
  # echo "New file path $NEWFILEPATH"
  
  # echo "Running ffmpeg."
  RES=$(ffmpeg -hide_banner -loglevel panic -i "$file" -ar 8000 "$NEWFILEPATH")
done 
echo "Done."

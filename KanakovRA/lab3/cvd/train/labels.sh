#!/bin/bash
script_dir=$(dirname "$0")
cd "$script_dir"
rm -rf labels.txt
touch labels.txt
for i in *; do 
  filename=$(basename $i)
  name=${filename%.*}
  if [[ "$name" =~ ^[c][a][t] ]]
  then
    echo "0" >> labels.txt
  else
    if [[ "$name" =~ ^[d][o][g] ]]
    then
      echo "1" >> labels.txt
    fi
  fi
done

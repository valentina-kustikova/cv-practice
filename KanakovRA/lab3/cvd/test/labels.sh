#!/bin/bash
script_dir=$(dirname "$0")
cd "$script_dir"
rm -rf labels.txt
touch labels.txt
for i in *;
  do filename=$(basename $i)
  name=${filename%.*}
  if [[ "$name" =~ ^[0-9]+$ ]]
    then if (("$name"<101))
      then echo "0" >> labels.txt
      else echo "1" >> labels.txt
    fi
  fi
done


#!/bin/bash

file="../data/1"

echo "Reading file: $file"

while read line
do
    echo "$line"
done < "$file"

echo "--------------------------"

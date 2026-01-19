#!/bin/bash

OUTDIR="data/raw/luna16"
mkdir -p $OUTDIR

URL="https://zenodo.org/record/3723295/files"

echo "Downloading annotations..."
wget -O "$OUTDIR/annotations.csv" "$FIRST_URL/annotations.csv?download=1"
wget -O "$OUTDIR/candidates.csv" "$FIRST_URL/candidates.csv?download=1"

echo "Downloading first three LUNA16 subsets!"

for i in {0..2}
do
    FILE="subset$i.zip"
    URL="$URL/$FILE?download=1"

    echo "Downloading $FILE ..."
    wget -O "$OUTDIR/$FILE" "$URL"
done

echo "Unzipping subsets..."
for i in {0..2}
do
    unzip "$OUTDIR/subset$i.zip" -d "$OUTDIR/subset$i"
done

echo "Dataset is downloaded!"
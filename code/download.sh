#!/bin/bash

OUTDIR="data/raw/luna16"
mkdir -p $OUTDIR

FIRST_URL="https://zenodo.org/record/3723295/files"
SECOND_URL="https://zenodo.org/records/4121926/files"

echo "Downloading LUNA16 subsets..."

for i in {0..6}
do
    FILE="subset$i.zip"
    URL="$FIRST_URL/$FILE?download=1"

    echo "Downloading $FILE ..."
    wget -O "$OUTDIR/$FILE" "$URL"
done

echo "Downloading annotations..."
wget -O "$OUTDIR/annotations.csv" "$FIRST_URL/annotations.csv?download=1"
wget -O "$OUTDIR/candidates.csv" "$FIRST_URL/candidates.csv?download=1"

echo "Unzipping subsets..."
for i in {0..6}
do
    unzip "$OUTDIR/subset$i.zip" -d "$OUTDIR/subset$i"
done

echo "First batch is downloaded!"


for i in {7..9}
do
    FILE="subset$i.zip"
    URL="$SECOND_URL/$FILE?download=1"

    echo "Downloading $FILE ..."
    wget -O "$OUTDIR/$FILE" "$URL"
done

echo "Unzipping subsets..."
for i in {7..9}
do
    unzip "$OUTDIR/subset$i.zip" -d "$OUTDIR/subset$i"
done
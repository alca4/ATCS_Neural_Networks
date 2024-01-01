javac BMP2OneByte.java
javac thingy.java
javac BGR2BMP.java

# mkdir Bins
# mkdir Activations
# mkdir Better_BMP
for i in `seq 1 5`
do
    # mkdir Bins/$i
    # mkdir Activations/$i
    # mkdir Better_BMP/$i
    for j in `seq 1 6`
    do
        echo "processing image $i $j"

        # touch Bins/$i/$i.$j.bin
        java BMP2OneByte BMP/$i/$i.$j.bmp Bins/$i/$i.$j.bin > aah.txt

        # touch CenteredBins/$i/$i.$j.bin
        # touch Activations/$i/$i.$j.txt
        java thingy Bins/$i/$i.$j.bin CenteredBins/$i/$i.$j.bin Activations/$i/$i.$j.txt > aah.txt

        java BGR2BMP gray 168 168 CenteredBins/$i/$i.$j.bin Better_BMP/$i/$i.$j.bmp > aah.txt
    done
done
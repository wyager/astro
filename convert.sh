for img in *.RAF; do 
    sips -s format tiff $img --out \
        $(exiftool  -iso -exposuretime -fnumber -T $img | sed $'s/\t/-/g').$img.tiff;
done

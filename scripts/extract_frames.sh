cd data

for i in {0..4}; do
    mkdir ./calib_challenge/labeled/$i
    ffmpeg -i ./calib_challenge/labeled/$i.hevc ./calib_challenge/labeled/$i/'%04d.jpg'
done

for i in {5..9}; do
    mkdir ./calib_challenge/unlabeled/$i
    ffmpeg -i ./calib_challenge/unlabeled/$i.hevc ./calib_challenge/unlabeled/$i/'%04d.jpg'
done
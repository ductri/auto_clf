echo "Clone ..."
git clone https://github.com/ductri/auto_clf.git

cd auto_clf

echo "Downloading data"
mkdir -p  source/main/data_for_train/output/train
wget http://213.246.38.101:10002/source/main/data_for_train/output/train/pool.csv \
    -O source/main/data_for_train/output/train/pool.csv
wget http://213.246.38.101:2609/source/main/data_for_train/output/train/positive_class_1.csv \
    -O source/main/data_for_train/output/train/positive_class_1.csv

mkdir -p  source/main/data_for_train/output/eval
wget http://213.246.38.101:10002/source/main/data_for_train/output/eval/pool.csv \
    -O source/main/data_for_train/output/eval/pool.csv
wget http://213.246.38.101:2609/source/main/data_for_train/output/eval/positive_class_1.csv \
    -O source/main/data_for_train/output/eval/positive_class_1.csv

mkdir -p  source/main/data_for_train/output/test
wget http://213.246.38.101:10002/source/main/data_for_train/output/test/pool.csv \
    -O source/main/data_for_train/output/test/pool.csv
wget http://213.246.38.101:2609/source/main/data_for_train/output/test/positive_class_1.csv \
    -O source/main/data_for_train/output/test/positive_class_1.csv

echo "Downloading vocab"
mkdir -p  source/main/vocab/output
wget http://213.246.38.101:2609/source/main/vocab/output/voc.pkl \
    -O source/main/vocab/output/voc.pkl

ln -s ./source /source

#!/bin/sh
mkdir data data/multi
cd data/multi
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -O real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip -O sketch.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip -O clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip -O painting.zip
#wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip -O quickdraw.zip
#wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip -O infograph.zip
unzip real.zip
unzip sketch.zip
unzip clipart.zip
unzip painting.zip
#unzip quickdraw.zip
#unzip infograph.zip

#wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip -O painting.zip
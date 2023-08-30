#!/bin/bash
gdown https://drive.google.com/uc?id=1AmsjJRJ095xjXn9KnYircBUsjj5P48WQ -O ./train_model/model.zip
unzip ./train_model/model.zip -d ./train_model
rm ./train_model/model.zip
rm -r ./train_model/__MACOSX/

gdown https://drive.google.com/uc?id=1LTFe0nVn9uQoq2p9T0iwr3HfMTlzeanQ -O ./data.zip
unzip ./data.zip
rm ./data.zip
rm -r ./__MACOSX/
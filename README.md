Please download the train and test data from `https://www.kaggle.com/c/dogs-vs-cats` and place the train and test images in `input/train` and `input/test` directories, respectively. We adopt our code from [Kaggle](https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification). 

To train the CNN model using float32.
```
Python lattice.py fp32
```

To train the CNN model using float16 (quantized version).
```
Python lattice.py fp16
```

Files:
- `lattice.py`: the top level funtion that eithor call float32 or float16 (quantized) 
- `mine_fp32.py`: Create a model to classfiy cats and dogs using float32
- `mine_fp16.py`: Create a model to classfiy cats and dogs using float16 (quantized) 
- `bn.py`: batch_normalization to support float16. We have implemeted a batchnormalization from scratch using python that will be published soon.
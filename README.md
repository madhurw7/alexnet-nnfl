# alexnet-nnfl
### An Alexnet based image classifier for the Tiny-ImageNet-200 Dataset

## Tiny-ImageNet-200 Dataset
ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. 

Tiny-ImageNet-200 is a subset of the ImageNet Dataset, and it contains 500 Training 64x64 Images each for 200 classes. ImageNet on the other hand has 256x256 images. The Class IDs of tiny-imagenet are based on WordNet IDs, same as ImageNet. Each Class is in a different folder of class ID (nxxxxxxxx).

The DataSet can be found over at [Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com/), or can be downloaded directly from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip).

## Instructions to run:
In case you don't have conda installed, head over to [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
After installing conda, enter the following in the Terminal:

Modify `prefix` in environment file to the location where you wish to install the environment

```
git clone https://github.com/madhurw7/alexnet-nnfl.git
cd alexnet-nnfl
conda env create -f environment.yml
jupyter notebook
```

Use the `classification-tut.ipynb` notebook and run it.
If you are running the notebook for the first time, set the `downloadDataset` variable to `True`, otherwise set it to `False`.

### Alternative
Alternativey you can run the `tiny-imagenet-classifier.py` file directly. Do keep in mind that the default value of `downloadDataSet` is set to `False`.
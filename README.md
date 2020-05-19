# alexnet-nnfl
### An Alexnet based image classifier for the Tiny-ImageNet-200 Dataset

## Tiny-ImageNet-200 Dataset
ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. 

Tiny-ImageNet-200 is a subset of the ImageNet Dataset, and it contains 500 Training 64x64 Images each for 200 classes. ImageNet on the other hand has 256x256 images. The Class IDs of tiny-imagenet are based on WordNet IDs, same as ImageNet. Each Class is in a different folder of class ID (nxxxxxxxx).

The DataSet can be found over at [Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com/), or can be downloaded directly from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip).

## Dataset Split
The Tiny-ImageNet data although contains seperate data for Training, Validation and Testing, but the labels for testing have not been provided and it was expected to be tested by submitting it to the Stanford CS231n team. 

We split the training data into a 80:20 split for training and validation. The validation folder was used to calculate the final accuracy instead of the testing folder data. 

## Model
We modified the AlexNet architecture to get better results on the tiny-imagenet database. We added 2 extra Convolutional Layers, to a total of **7 Convolutional Layers** and **2 Fully Connected Layers**. We used Batch Normalisation and Max Pooling to improve our results. The trained model's parameters are available in this repository as `tinynet.h5`.

## Results
We were able to achieve a **top 1 accuracy rate** of **44.89%** and a **top 5 accuracy rate** of **71%**.

## Instructions to run:
In case you don't have conda installed, head over to [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

After installing conda, enter the following in the Terminal:

```
git clone https://github.com/madhurw7/alexnet-nnfl.git
cd alexnet-nnfl
conda env create -f environment.yml
jupyter notebook
```

Modify `prefix` in environment file to the location where you wish to install the environment

### Training
Use the `tiny-net.ipynb` notebook and run it.
If you are running the notebook for the first time, set the `downloadDataset` variable to `True`, otherwise set it to `False`. The code is written for **Tensorflow 2.1.0**, if you are using online platforms like Google Colab or Google Cloud Platform, you will have to run the following first:

```
%pip install tensorflow==2.1.0
!pip install pyyaml h5py
```

Restart the Kernel for the changes to take effect. You can check Tensorflow version by printing `tf.__verssion__`

### Test Data Generation
Run the `test-data-generator.ipynb` file to generate the `valDataComplete.npz` file which is required for evaluation. Takes in class names from `wnids.txt` and creates input and output numpy arrays which can be fed into the `model.predict()` method.

### Evaluation
Run the `tiny-net-evaluate.ipynb` file, given both `valDataComplete.npz` and `tinynet.h5` are present in the root of the repository. Feeds the test data to the saved model and determines final test top-1 and top-5 accuracy.
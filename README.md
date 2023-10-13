# Thesis - Deepfake Detection

This repo contains source files for my Thesis titled: "Compression-robust Deepfake detection via Deep Learning
", submitted at the Electrical and Computer Engineering Dept. of Aristotle 
University of Thessaloniki, Greece.

## Thesis file
A draft of the Thesis document can be found in `Deepfake_Thesis.pdf` (Greek version)

## Requirements
The project is based in Python; the requirements for this project can be found
in `requirements.txt`.

The FaceForensics++ dataset and the MTCNN face detection Neural Network are also utilized.
Links and instructions on how to obtain them can be found below.

## Dataset
The FaceForensics++ dataset is used in this project, which can be found in 
[this](https://github.com/ondyari/FaceForensics) Github repo.

It contains 1000 real videos and 5000 fake/manipulated Deepfake videos in three different quality/ compression 
levels (Raw, HQ, LQ, a total of 18000 videos, excluding the additional DFD dataset provided).

You can apply for access to this dataset 
[here](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform). 
After being granted access, follow the instructions provided, as outlined 
[here](https://github.com/ondyari/FaceForensics/tree/master/dataset), in order to download the datasets and split
the dataset in train, validation and test parts.

### Frame extraction
For the face detection process, the MTCNN face detction Neural Network from facenet-pytorch is used.
The package can be found in [this](https://github.com/timesler/facenet-pytorch) repo.

After downloading the dataset files for a particular quality, `frame_extraction.py` can be used to sample the videos, 
detect the main face, crop and store the frames for each video, provided an index file containing all downloaded files
for this quality is created.

### Creating Extra Quality/ Compression levels
After having downloaded the uncompressed (c0) videos of the FF++ dataset, videos of any Constant Rate Factor (CRF)
can be created using `create_quality.py`, provided an index file containing all downloaded files
for this quality is created.


## Models
Source code as well as some saved models (weights) for the Models described on the Thesis are provided in this repo,
in the `models` folder.

There are five models used in this Thesis, namely `base_model`, `similarity_model`, `adversarial_model`, 
`similarity-adversarial_model` and `ensemble_model`.

Each model contains functions for training, loading and testing the models which can be used after downloading the 
FaceForensics++ dataset.

## Indexes and dataset file structure
The indexes provided are only for demonstrative and aiding purposes, as the dataset
is not part of this repo. It is recommended that indexes are reconstructed as needed.

The index files used for the models' dataloaders, suppose a file structure similar to the one
outlined below:

```
dataset
├───ff++
│   ├───Deepfakes
│   │   ├───c0
│   │   ├───c23
│   │   └───c40
│   ├───Face2Face
│   │   ├───...
│   ├───FaceShifter
│   │   ├───...
│   ├───FaceSwap
│   │   ├───...
│   ├───NeuralTextures
│   │   ├───...
│   └───Real
│       └───...
└───videos
    ├───c0
    │   ├───Deepfakes
    │   ├───Face2Face
    │   ├───FaceShifter
    │   ├───FaceSwap
    │   ├───NeuralTextures
    │   └───Real
    ├───c12
    │   ├───...
    ├───c23
    │   ├───...
    ├───c31
    │   ├───...
    └───c40
        └───...
```

## Help
If you have any questions, please contact me at [vterzoglou@gmail.com](mailto:vterzoglou@gmail.com).
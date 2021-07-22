
# Image Search Engine for Digital History: A deep learning approach

This repository contains the source code and documentation of the *Image Search Engine for Digital History: A Deep Learning approach* project. It is part of the [Engineering Historical Memory](https://engineeringhistoricalmemory.com/About.php) research, contributing to a multilingual and transcultural approach to decode-encode the treasure of human experience and transmit it to the next generation of world citizens.

More information about the research project can be found [here]().
## Demo

TODO

## Documentation

This section contains the documentation of the source code of this repository. Some content is based on external repositories and referenced accordingly.

### Repository structure

    .
    ├── input                           # Folder containing the input information (see `Input`)
    │   │
    │   ├── _haystack                   # Folder containing the database images
    │   ├── _needle                     # Folder containing the images to be found
    │   ├── dataset_file.csv            # CSV file containing the references to the dataset
    │   └── generate_dataset_file.py    # Python executable to generate the CSV file
    │      
    ├── lib                             # Folder containing the library files (see `Libraries`)
    ├── models                          # Folder containing the model files (see `Models`)
    ├── output                          # Folder containing the output files (see `Output`)
    ├── .gitignore                      
    ├── main.py                         # Executable Python code for running the engine
    ├── LICENSE                         # License information
    ├── README.md
    └── requirements.txt                # dependencies (see `Requirements`)

### Requirements
This repository is Python based and has the following dependencies:
- numpy
- torch
- Pillow
- tqdm
- pandas
- scipy
- pydegensac
- opencv-python

The dependencies can be easily installed using the package installer [pip](https://pypi.org/project/pip/) by `pip install -r requirements.txt`

### Parameters

Directory and file specification

```python
BASE = "./"                             
OUTPUT = BASE + "output/"
INPUT = BASE + "input/"
MODULE_FILE = BASE + 'models/d2_tf.pth'
```

D2-Net parameters (more information [here](https://github.com/mihaidusmanu/d2-net))

```python
PREPROCESSING = 'caffe'                                 # Pre-processing method (None, caffe, or torch)
USE_RELU = True                                         # Make use of the Rectified Linear Unit
OUTPUT_TYPE = 'npz'                                     # Extraction output file extension (npz or mat)
MULTISCALE = True                                       # Use multiscale or not

MAX_EDGE = 1600                                         # Max edge size (width or height) in pixels

MAX_SUM_EDGES = 2800                                    # Max sum of edges (width + height) in pixels
   
OUTPUT_EXTENSION = '.d2-net'                            # Extracted file extension
```

This threshold determines the engine's decision on whether a match is found. Read the thesis paper for more information.

```python
THRESHOLD = 21                                          # Inlier threshold
```

### Input
To run the search engine, a interpretable dataset file needs to be in place. This can be done using the `generate_dataset_file.py`, `_haystack`, and `_needle` folders found in the `input` folder. The idea is to group similar images (jpg, jpeg, and png format) by placing these in the same sub-folder in the `_haystack` folder. Then, in the `_needle` folder, a similarly named sub-folder is placed, containing the needle(s) that match with the corresponding `_haystack` images. When finished, run the `generate_dataset_file.py` file to generate the `dataset_file.csv`.

The dataset already visible in the `dataset_file.csv` can be downloaded [here](https://drive.google.com/file/d/1N_aGtz41ty6dTBoRzF6wMF1L0f2noBzY/view?usp=sharing). The images in this dataset have been obtained from Engineering Historical Memory and contain images from the web with no commercial purposes.

### Libraries

The libraries (`lib` folder) are specified in the [D2-Net](https://github.com/mihaidusmanu/d2-net) repository.

### Models

The pre-trained models, made available by the  [D2-Net](https://github.com/mihaidusmanu/d2-net) authors, are put in the `models` folder. More information about these can be found in the  [D2-Net](https://github.com/mihaidusmanu/d2-net) repository.

### Output

The output folder contains the resultant dataset file and images displaying the algorithm matching.

## Usage
The image search engine should be ran on a server computer with GPU support (~12GB VRAM if multiscale enabled).

To run the engine, make sure the input is correctly configured, navigate to the directory, and run `python3 main.py`.
## Authors

The authors of this repository are Mathijs van Geerenstein, Philippe van Mastrigt, and Laurens Vergroesen. The content of this repository has been made possible by other developers and work. Relevant sources for the code can be found in the [Sources](#references) section. 

The research and its references are extensively described in the available [research paper](http://resolver.tudelft.nl/uuid:f1a2902b-14be-416c-ae1a-ce4f179a0425).


## Acknowledgements

This research is made possible by Dr. Justin Dauwels, Dr. Andrea Nanetti, and the collaboration of Delft University of Technology (TUDelft) with Nanyang Technological University Singapore, School of Art, Design and Media (NTU-ADM).

## Sources
This repostiory contains code from the [D2-Net](https://github.com/mihaidusmanu/d2-net) repository, the [OpenCV]() repository, and the [Pydegensac](https://github.com/ducha-aiki/pydegensac) repository. The D2-Net code is used for local feature extraction, the OpenCV code is used for matching, and Pydegensac is used for verifying good matches by calculating inliers. 

## License
This repository is licensed under [(CC) BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Except where otherwise noted, content is licensed under Attribution-NonCommercial-NoDerivatives 4.0 International

![](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png)
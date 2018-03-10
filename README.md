# Computer-Vision-Exercise

Small programming task to find and highlight the variability in the field by identifying and outlining the areas of irregularity in the field.

## Getting Started

1. Download [detect_abnormal.py](https://github.com/JimmeeX/Computer-Vision-Exercise/blob/master/detect_abnormal.py)

2. In the same directory as the code, make a new directory called "Input", and a new directory called "Output"

```
mkdir Input Output
```

3. Load the .tif images for analysis in the "Input" directory (see [Data](https://github.com/JimmeeX/Computer-Vision-Exercise/tree/master/Data) for reference)

4. Run the code:

```
python detect_abnormal.py Input/ Output/
```

5. Find results stored in the "Output" folder. (see [Results](https://github.com/JimmeeX/Computer-Vision-Exercise/tree/master/Results) for reference)



To see how the code works, read the jupyter notebook file - [detect_abnormal.ipynb](https://nbviewer.jupyter.org/github/JimmeeX/Computer-Vision-Exercise/blob/master/detect_abnormal.ipynb). 

### Prerequisites

What things you need to install the software and how to install them

```
python==3.6.4

numpy==1.14.0

opencv-python==3.4.0.12

Pillow==5.0.0

matplotlib==2.1.2

scikit-image==0.13.1
```

### Sample Results

CCCI Image

![alt text](https://github.com/JimmeeX/Computer-Vision-Exercise/blob/master/Results/20171219T000000_HIRAMS_PLN_ccci_gray.png)

MSAVI Image

![alt text](https://github.com/JimmeeX/Computer-Vision-Exercise/blob/master/Results/20171212T000000_HIRAMS_PLN_msavi_gray.png)

NDVI Image

![alt text](https://github.com/JimmeeX/Computer-Vision-Exercise/blob/master/Results/20171212T000000_HIRAMS_PLN_ndvi_gray.png)

NDRE Image

![alt text](https://github.com/JimmeeX/Computer-Vision-Exercise/blob/master/Results/20180119T000000_TETRA-_PLN_ndre_gray.png)

### Improving the Results

The hyperparameters in detect_abnormal.py in the beginning of the file can be fine-tuned to produce better results.

## Authors

* **James Lin** - [JimmeeX](https://github.com/JimmeeX)
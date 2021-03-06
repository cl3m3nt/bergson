# Bergson Astro Pi project

This repository hosts the project done by Bergson Highschool as part of Astro Pi Challenge 2021.
The Bergson team worked on building an Artificial Intelligence model predicting NO2 polution from NDVI pictures.

<img width="595" alt="Screenshot 2021-02-12 at 16 05 40" src="https://user-images.githubusercontent.com/8882133/107785033-64bd4600-6d4c-11eb-9976-c3132764dd0d.png">

In addition of detecting NO2 polution to help protecting Earth, we wanted to evaluate Artificial Intelligence opportunity to be run on Astro Pi.
To do so, we built two Deep Learning classifier models, a simple Convolutional Neural Network and a Mobilenetv2 Neural Network.
As part of our project, we wanted to validate the feasibility of running real time inference on Astro Pi limited Hardware.
We successfully ran inference with both simple Conv2D model and an optimized version of Mobilnetnetv2 with TFlite on both Desktop and FlightOS.

## Pre-Requesite
Make sure to install Tensorflow 1.14 and Keras 2.2.5 modules as they are mandatory per Astro Pi challenge guidance.
We used a specific astropi conda environment to reproduce Astro Pi contraints and debug on local PC as well as raspberry pi.
You can install all pre-requesite libraries with:
```bash
pip install -r requirements.txt
```
More information about Astro Pi libraries and HW can be found [here](https://projects.raspberrypi.org/en/projects/code-for-your-astro-pi-mission-space-lab-experiment/2)


## Training Neural Network models
The training script [here](https://github.com/cl3m3nt/bergson/blob/master/src/training.py) will train both a 2D Convolutional Network as well as a Mobilenetv2 based Neural Network using Transfer learning.
We reached with our limited Dataset 0.9512 accuracy with Conv2D after 10 epochs  and 0.8537 accuracy with Mobilenet after 20 epochs on training data. Because of Astro Pi challenge short timing we could not invest as much as we wanted on building a robust dataset with more data and both training and validation data. As a first shot, we hope that using Mobilenetv2 with our limited data would anyway provide some interesting result.

```bash
python3 training.py
```

## Predicting with Neural Network models
The main script [here](https://github.com/cl3m3nt/bergson/blob/master/src/main.py) will do inference and run only on Astro Pi hardware as it requires a Raspberry Pi camera Hardware. Data we will collect during experiment will allow us adress the model validation challenge, as we will leverage it to measure how good our model made prediction. 
The default version of the script will use Mobilenetv2 architecture, as we thought it more robust than simple Conv2D model.
To allow Mobilenetv2 architecture to effectively run on Astro Pi, we use the TFLite converter to make sure the HW can process inference.
Even though it's challenging, we successfully ran the experiment on Flight OS for 3 hours, making about 10 x inferences per minute at 98% CPU usage.
In case it would not run on ISS actual Astro Pi, we can fall back to using Conv2D model instead which is less computational heavy.
To do so, comment Mobilenetv2 related lines in main function and uncomment Conv2D related ones.


```bash
python3 main.py
```

Before running main.py, make sure you have copied/moved the models files [here](https://github.com/cl3m3nt/bergson/tree/master/models) in the same directory than main.py as they are required to be loaded for the script to successfully run.

Otherwise, you can download only bergon.zip file [here](https://github.com/cl3m3nt/bergson/blob/master/bergson.zip) and unzip it.
To use the training.py script, you can download the dataset [here](https://storage.googleapis.com/bergsondataset/dataset.zip) within your extracted bergson folder. Unzip dataset then you should be able to train again the Neural Networks.


## Results from ISS
Data back from ISS allows to evaluate the quality of the first version of our model.
A selection of x245 pictures available [here](https://storage.googleapis.com/bergson_iss/pictures.zip) has been used to compare our AI model NO2 prediction vs ESA [Sentinel-5P tropospheric satellite](https://maps.s5p-pal.com/) NO2 values.
We post-processed this data to finally output the Bergson team results anlysis for Astro Pi challenge phase 4. Our full report is available [here](https://github.com/cl3m3nt/bergson/blob/master/results/Astro_Pi_Mission_Space_Lab_Bergson_Report.pdf)
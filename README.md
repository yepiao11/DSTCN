# DSTCN
This is an implementation of DSTCN:[Exploiting Dynamic Spatio-Temporal Correlations For Origin-Destination Demand Prediction]

## Introduction
We design a novel Dynamic Spatio-Temporal Correlation Network (DSTCN) to address the OD demand prediction problem. Our model comprises
three modules. The first module fully considers the OD demand tendency correlations to capture the OD demand changes in terms of
the OD and DO (Destination-Origin) orientations; the second module grabs the dynamic spatio-temporal characteristics of each area;
and the third module integrates temporal knowledge to make next-step predictions. DSTCN can capture the complex spatial-temporal correlations between the areas in the OD demand prediction. 

## Network Structure
![image](https://github.com/yepiao11/DSTCN/blob/main/figures/DSTCN_Framework.png)

**Framework of DSTCN**, which comprises three modules: the first module, (Glstm2D) fully considers the OD demand tendency correlations; the second module (Simformer) captures the dynamic spatial features of each area; and the third module (F-GRU) integrates the dynamic demand correlations extracted from Glstm2D and Simformer, and makes next-step predictions via a well-designed multi-head temporal convolution layer.

## Results
![image](https://github.com/yepiao11/DSTCN/blob/main/figures/result_nyc.png)
<br />
<p align='center'><b>Experimental Results of New York Taxi Dataset in Manhattan in 2018 and 2019<b></p>

![image](https://github.com/yepiao11/DSTCN/blob/main/figures/result_hz.png)
<br />
<p align='center'><b>Experimental results of Hangzhou Metro dataset in 2019<b></p>

![image](https://github.com/yepiao11/DSTCN/blob/main/figures/the%20result%20of%20OD%20prediction.png)
<br />
<p align='center'><b>Visual comparison of OD distribution prediction results between DSTCN and other OD models<b></p>


## Environment
* Python  3.7.11
* Cuda 10.2
* PyTorch 1.8.1
* Numpy 1.21.6

## Dataset
Step 1: Download the processed dataset from [Baidu Yun](https://pan.baidu.com/s/1XhvSA2qoJOLr72SQ9H4WpQ) (Access Code:dyov).    
Step 2: Put them into ./data directories.
## Train command
python train_model_pre1.py
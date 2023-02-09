# ocr-handwritten
An end-to-end model development for handwritten OCR 

Experiment Tracking (here)[https://app.clear.ml/projects/d60f182ac5104605afd7f1c45ff2a927/experiments/f2fb3eb021f8408b9da26fd91ac65c20/output/execution]

# Requirements
Please make sure that you have docker installed on you device.

# Weights
Download craft model from [here](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) and place it to `scripts/model/`

Download trocr model from [here](https://drive.google.com/drive/folders/1TR_Hrre_SqXKICyWTUeG799NG-OZVnw7?usp=sharing) and extract the model to `scripts/model/` and rename as `trocr-finetuned-augmented`


## Run the program
First, build docker image
`docker-compose build`

the, run the container
`docker-compose up`
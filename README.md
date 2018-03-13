# Training A Recurrent Neural Network to Predict Sentiment

## Setup
You will need the following dependencies and languages to run this repository
### Languages
* Python 3.5.4

### Libraries
* bs4==0.0.1
* jupyter==1.0.0
* matplotlib==2.0.2
* numpy==1.13.1
* scipy==0.19.1
* tensorflow==1.3.0

You can also install these libraries through the `requirements.txt` file using `pip install -r requirements.txt`

## Scraping and Organizing Data
To save space on the repository and hide unnecessary folders and files, you need to set up your training and testing folders before running the RNN.

### Scraping
You should not need to scrape data again unless you're adding new raw data. Instructions for this will come later.

### Organizing
In the home project directory simply run `python organize.py yourfilepath`. Doing so will create the directories if they do not exist and then place appropriate files in each directory.

## Training
Run `python net.py`. More documentation to come.



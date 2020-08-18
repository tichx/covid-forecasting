## How to forecast COVID-19 trends using PyTorch

> This tutorial is NOT trying to build a model that predicts the Covid-19 outbreak/pandemic in the best way possible. This is an example of how you can use Recurrent Neural Networks on some real-world Time Series data with PyTorch. Hopefully, there are much better models that predict the number of daily confirmed cases.

Time series data captures a series of data points recorded at (usually) regular intervals. Some common examples include daily weather temperature, stock prices, and the number of sales a company makes.

Many classical methods (e.g. ARIMA) try to deal with Time Series data with varying success (not to say they are bad at it). In the last couple of years, [Long Short Term Memory Networks (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) models have become a very useful method when dealing with those types of data.

Recurrent Neural Networks (LSTMs are one type of those) are very good at processing sequences of data. They can "recall" patterns in the data that are very far into the past (or future). In this tutorial, you're going to learn how to use LSTMs to predict future Coronavirus cases based on real-world data.

## Instructions

This code has only been tested using Python 3.7.3.  Training has been tested on GCE machines with 8 V100s, running Ubuntu 16.04, but development also works on Mac OS X.

### Installation

- Install [pipenv](https://github.com/pypa/pipenv#installation).

- Install [tensorflow](https://www.tensorflow.org/install/gpu):  Install CUDA 10.0 and cuDNN 7.6.2, then `pipenv install tensorflow-gpu==1.13.1`.  The code may technically run with tensorflow on CPU but will be very slow.

- Install [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install)

- Clone this repo.  Then:
  ```
  pipenv install
  ```

- (Recommended) Install [`leaderboard_toolkit`](https://github.com/horovod/horovod#install) to track experiment metadata. Make sure to fully install packages from 'requirements.txt'

### Sample 

```python
# Import metadata tracking
from leaderboard_toolkit import visualize, track


def train(x_train, y_train, x_test, y_test, copy_train):
    # Run your code
    model = build_model_graph()
    model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
   # Track training
   leaderboard_toolkit.track(model)

train(x_train, y_train, x_test, y_test)

```


### Results
Experiment metadata are stored in folder directory
```
./metrics/
```


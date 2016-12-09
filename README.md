# RNN-Tagger
build a recurrent neural network for part-of-speech tagging problem


## requirements
tensorflow(>r0.10)
nltk

## model
I use a bidirectional lstm-rnn to finish this task as well as a bucket queue
The data is from nltk so if you want to run this model, make sure you have
already installed nltk

## benchmark
after training for 10 minutes in a single tesla K20m, this model achieved
a peak accuracy about 92%

## TODO
add multi gpu train


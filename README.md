# Neural-Machine-Translation
Neural Machine Translation by Seq2Seq Model with Attention layer

This is a sample neural machine translation project with converts english to french. Don't expect it to perform as good as google translate because it had been trained on very small dataset. You can train on big dataset and I'm sure it will perform good.

## Overview
* It is based on Seq2Seq Rnn model with attention layer.<br/>
* Seq2Seq have two main parts, the encoder model and the decoder model. <br/>
* The decoder model can be purely Seq2Seq or with a custom attention layer that improves the accuracy of model.

## Dependencies
* Python 3+
* Keras with tensorflow backend
* nvidia Gpu (for training purpose as it use CuDNNLSTM layer that is accelerated by CuDNN library by nvidia)
* Numpy

## How to use
1. Fork this repo
2. Download the dataset from <a href='https://github.com/susanli2016/NLP-with-Python/tree/master/data'>here</a> .
3. Download the GloVe Word embeddings from <a href='http://nlp.stanford.edu/data/glove.6B.zip'> here</a>.
4. Save both data and GloVe embeddings in ```data``` folder.
5. If training, make changes in file ```utils/config.py``` if you want.
6. Use the ```train.ipynb``` notebook for training.
7. If using for test-predictions, download the weights from <a href='https://drive.google.com/open?id=1x47sdloj6Ah6F7F7YvPZLRD8Pie7zsu8'>here</a> and save it in ```weights``` folder.
8. Use ```test-translations.ipynb``` notebook.

## Referneces
1. check out Deep NLP course by <a href='https://www.udemy.com/deep-learning-advanced-nlp/'> Lazy Programmer Inc. </a>
2. For actual code check his <a href='https://github.com/lazyprogrammer/machine_learning_examples'> repo</a>.
3. Also check out <a href='https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/'> this </a> cool article about Seq2Seq model with attention layer.
 


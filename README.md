# Medical Images Project
In this repository, we show how to build medical image systems with the CASL projects. We showcased that [Texar](https://github.com/asyml/texar-pytorch/) as
a modularized, versatile, and extensible toolkit can be efficiently used for text generation tasks. Moreover, we used [Forte](https://github.com/asyml/forte/) to conduct data preprocessing tasks.

## Introduction
We utilized a hierarchical LSTM structure to generate the medical image report. There are mainly six modules: a CNN feature extractor, a multi-label classifier (MLC), a SemanticTagGenerator, a CoAttention module, a Sentence LSTM and a Word LSTM.

It first extracts visual features from the input images.
The MLC is then used to predict the probabilities of active tags. The
SemanticTagGenerator selects the top K tags with the highest probabilities
and generate corresponding semantic features. The CoAttention module then
extracts context vector from both the visual and semantic features
based on the previous state of the Sentence LSTM.

By conditioning on the context vectors, a two 2 layered hierarchical LSTM are proposed to generate the actual paragraph. Specifically, the first layer is the Sentence LSTM that generate "topic" for each sentence by taking as input the context vector. The "topic" is passed to the Word LSTM and generate the content for each sentence. Moreover, the Sentence LSTM also learns to control the continuation of sentence generation.

## How to run

Step 1, data preprocessing for the image and text data. Split them into `train`, `val` and `test`.
```bash
python split_train_val_test.py

python textdata_preprocessor.py
```
Step 2, specify the paths to image and text data in `config.py`. 

Step 3 (Optional), pretrain the MLC head, and specify the path to the `pretrained weights` in `config.py`. Note that you can determine whether to train the encoder in `config.py` by setting the value for `dataset["mlc_trainer]["train_encoder"]`.

```bash
python train_mlc.py
```

Step 4, train the two layer LSTM and obtain the inference performance.
```bash
python train_lstm.py
```

## Performance
The results are obtained by using the default hyper-parameters listed in `config.py` without pretraining the MLC.

| Mode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | --- | --- | --- | --- |
| Reproduce | 0.554 | 0.231 | 0.136 | 0.067 |
| Paper | 0.517 | 0.386 | 0.306 | 0.247 |

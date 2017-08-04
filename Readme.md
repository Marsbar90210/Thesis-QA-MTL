##Thesis-QA-MTL

This repository contains the code used for my master thesis at university of copenhagen,
Multitask learning for question answering using neural networks

To run the code:
Python <modelfile> <ver> <qas> <iterations>

Here <modelfile> is one of the files prefixed with model in Code/
<ver> is one of following data versions: en, en-10k, hn-10k, en/hn, en-10k/hn, en/hn-10k, en-10k/hn-10k
<qas> is a list of questions separeted by , represented as qa1 upto qa20
they can further be concatenated by . to run them in a mtl fashion if the chosen model supports it
<iterations> specifies how many iterations to run of each experiment

More information can be found in the thesis.
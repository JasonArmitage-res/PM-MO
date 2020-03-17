# PM+MO

## System

PM+MO is a framework for multilabel classification with multimodal fusion. Text and image embeddings are fused and mapped to a label (or subset of labels). The system is trained using multiple objectives and multimodal fusion is conducted with variational inference. 

This system started as a port of the Gated Multimodal Unit (GMU) architecture (Arevalo et al) from Theano to PyTorch. Additional information on this research can be found here:
https://github.com/johnarevalo/gmu-mmimdb


## Setup

Dependencies are listed in the requirements.txt file. Instructions and documentation for Pyro PPL are located here:
https://pyro.ai/

Instructions for running the code:

```bash
python run.py multimodalimdb.h5 initial_values.json
```

## Dataset
Multilabel classification is conducted on the MM-IMDb dataset. Details can be found in the GMU Github repo linked above.

A copy of the dataset used in this research can be downloaded from here:
http://lisi1.unal.edu.co/mmimdb/multimodalimdb.hdf5

## License
[MIT](https://choosealicense.com/licenses/mit/)
To clarify, *original model* is the one provided in the NMT experiment. *New model* is the implementation of Fan et al, 2017. 

# Experiment 1. Identical setups on NMT
The new model was adjusted to match the original model, so both models were trained with the same architecture and optimizer on the task of neural machine translation from German to English: 
* CNN-based encoder and decoder
* 8 layers each in encoder and decoder
* Encoder and decoder connected through attention
* No self-attention
* Emb dim: 256, Hid dim: 512
* Total number of parameters in *original* model: 37,351,685
* Total number of parameters in *new* model: 37,351,685
* Adam optimizer, standard learning rate

The dataset used for this experiment was Multi30K, consisting of pairs of parallel sentences in English and German. Dataset statistics are as follows: 

* Train set: 29000, validation set: 1014, test set: 1000
* Vocab size, German (=input dim): 7855
* Vocab size, English (=output dim): 5893
* **German**:
* * Mean sentence length (in tokens): 12.439689655172414
* * Minimum length: 1
* * Maximum length: 44
* **English**:
* * Mean sentence length (in tokens): 13.109862068965517
* * Minimum length: 4
* * Maximum length: 41

Both models were trained for 10 epochs, and the model with best performance on validation set was saved. Performance of best model on the test set:
* Original: | Test Loss: 1.862 | Test PPL:   6.437 |
* New: | Test Loss: 1.637 | Test PPL:   5.139 |

Some sample responses, as produced by the two models (without teacher forcing):  
* Ground truth: ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
*  Original model inference: ['a', 'little', 'girl', 'climbing', 'in', 'a', 'playhouse', 'out', 'wood', '.', '<eos>']
* * New model inference: ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.', '<eos>']
* Ground truth: ['two', 'men', 'are', 'at', 'the', 'stove', 'preparing', 'food', '.']
* * Original model inference: ['two', 'men', 'stand', 'on', 'the', 'stove', 'preparing', 'food', 'and', 'preparing', 'food', '.', '<eos>']
* * New model inference: ['two', 'men', 'are', 'standing', 'on', 'the', 'stove', 'preparing', 'food', '.', '<eos>']

# Experiment 2. Text summarization architecture tested on NMT
For this experiment, the task remains the same: translation from German to English. However, this time the implementation used for the controllable abstractive summarization task is compared with the original NMT model. Therefore, the model architecture for *original* remains the same as in experiment 1, but the model achitecture *new* as adjusted with the following modifications:
* Number of layers reduced from 10 to 8
* Self-attention added
* Embedding dimension increased from 256 to 340
* SGD optimizer used with learning rate 0.2 decreased when validation loss stops decreasing
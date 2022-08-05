# METRO

__METRO: Efficient Denoising Pretraining of Large Scale Autoencoding
Language Models with Model Generated Signals__ by Microsoft on Apr 2022


## Improvements Over ELECTRA

__Relative Position Embeddings__: Similar to T5 and DeBERTa, the model uses relative position bins as opposed to absolute position embeddings of ELECTRA.

__Large Vocabulary__: Similar to DeBERTa, the model uses a cased sentence piece BPE vocabulary of 128K tokens. Large voabulary improves the capacity without effecting inference speed much.

__Simplified CLM__: Only applies CLM to masked positions.

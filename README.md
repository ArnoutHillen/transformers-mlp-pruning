The changes to the BERT model can be found in this [file](https://github.com/ArnoutHillen/transformers-mlp-pruning/blob/master/src/transformers/models/bert/modeling_bert.py).

## Steps to prune a BERT model.

- Install via pip.
```bash
pip install git+https://github.com/ArnoutHillen/transformers-mlp-pruning.git@master
```

- Create a tokenizer and a model.
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```


- Prune individual neurons by providing a dictionary that has for each layer the neurons to keep as a list (e.g., for layer 3 it would be {3:[0,1,2]}).
```python
model.bert.prune_all_mlp_neurons_except(neurons)
```


- Prune feedforward networks (e.g., in layer 8 and 9).
```python
model.bert.encoder.layer[8].apply_ffn = False
model.bert.encoder.layer[9].apply_ffn = False
```


- Or prune entire layers (attention + feedforward + add and norm connections) by providing a tuple of the layer indices.
```python
model.remove_layers((10,11))
```

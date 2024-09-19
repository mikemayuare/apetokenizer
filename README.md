# APE Tokenizer

**APE Tokenizer (Atom Pair Encoding Tokenizer)** is a tokenizer designed to handle SMILES and SELFIES molecular representations. It works similarly to BPE (Byte Pair Encoding), while ensuring that tokens preserve chemical information, making it ideal for molecular data. This tokenizer is fully compatible with the Hugging Face `transformers` library and can be easily integrated into any model that uses tokenizers.

## Features

- **Hugging Face `transformers` compatible**: Integrates seamlessly with models in the `transformers` library.
- Tokenizes both SMILES and SELFIES representations.
- Adds and manages special tokens such as `<pad>`, `<s>`, `</s>`, `<unk>`, and `<mask>`.
- Supports vocabulary management, tokenization, padding, and encoding with PyTorch and TensorFlow.
- Trains a vocabulary from a corpus of molecular sequences.
- Provides saving/loading functionalities for vocabularies and tokenizer states.

## Installation

To use the APE Tokenizer in your project, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/ape-tokenizer.git
cd ape-tokenizer
```

You’ll also need to install the Hugging Face `transformers` library:

```bash
pip install transformers
```

## Usage

### Using APE Tokenizer with `transformers`

You can easily integrate the APE Tokenizer with Hugging Face's `transformers` library. Here’s an example of tokenizing a SMILES sequence and using it with a model:

```python
from ape_tokenizer import APETokenizer

# Initialize the tokenizer
tokenizer = APETokenizer()
tokenizer.load_vocabulary("path_to_vocabulary.json")
# Example SMILES string
smiles = "CCO"

# Tokenize the SMILES string
encoded = tokenizer(smiles, add_special_tokens=False)

# Hugging Face compatible input
from transformers import AutoModelforSequenceClassification

# Load a pre-trained model (for demonstration purposes)
model = AutoModelforSequenceClassification.from_pretrained("mikemayuare/SELFY-BPE-BBBP")

# Use the tokenized input in the model
outputs = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])

print(outputs)
```

### Loading a Vocabulary

To load a saved vocabulary from a JSON file, use the `load_vocabulary` method:

```python
from ape_tokenizer import APETokenizer

# Initialize the tokenizer
tokenizer = APETokenizer()

# Load the vocabulary from a JSON file
tokenizer.load_vocabulary("path_to_vocabulary.json")

# Now the tokenizer is ready to tokenize sequences
smiles = "CCO"
encoded = tokenizer(smiles, add_special_tokens=False)
print(encoded)
```

### Tokenizing a DatasetDict
```python
def preprocess(examples):
    example = tokenizer(
        examples["selfies"],
        add_special_tokens=False,
    )

    return example


tokenized_dataset = dataset.map(preprocess)
```

### Saving and Loading a Vocabulary

To save the current tokenizer vocabulary and state to a JSON file:

```python
# Save the vocabulary and training state to JSON
tokenizer.save_vocabulary("path_to_save_vocabulary.json")
```

### Training a New Tokenizer

You can train the tokenizer from a corpus of molecular sequences (SMILES or SELFIES) using the `train` method. The trained tokenizer can then be saved and reused with `transformers`.

```python
# Example corpus of molecular sequences
corpus = ["CCO", "C=O", "CCC", "CCN"]

# Train the tokenizer with the corpus
tokenizer.train(corpus, max_vocab_size=5000, min_freq_for_merge=2000, save_checkpoint=True, checkpoint_path="./checkpoints")

# Save the trained vocabulary to a file
tokenizer.save_vocabulary("trained_vocabulary.json")
```

### Padding and Attention Mask Generation

Compatible with `DataCollatorWithPadding` class.

## Compatibility with Hugging Face `transformers`

APE Tokenizer is fully compatible with the `transformers` library. Here are the key methods to integrate APE Tokenizer with your transformer models:

### Integration Methods

- **`__call__(text, padding=False, max_length=None, add_special_tokens=False, return_tensors=None)`**: Tokenizes the input text and returns the tokenized sequences (compatible with models in the `transformers` library).
  
- **`train(corpus, max_vocab_size=5000, min_freq_for_merge=2000, save_checkpoint=False, checkpoint_path="checkpoints", checkpoint_interval=500)`**: Trains the tokenizer on a given corpus of molecular sequences.

- **`save_vocabulary(file_path)`**: Saves the current vocabulary to a JSON file.

- **`load_vocabulary(file_path)`**: Loads a vocabulary from a JSON file.

- **`encode(text, padding=False, max_length=None, add_special_tokens=False)`**: Encodes a given text into a sequence of token IDs.

- **`pad(batch, padding=False, return_tensors=None)`**: Pads a batch of sequences to a common length for model input.

- **`convert_tokens_to_ids(tokens)`**: Converts a list of tokens to their respective IDs.

- **`convert_ids_to_tokens(token_ids)`**: Converts a list of token IDs back to tokens.

## Example

Here’s a complete example that shows how to train the tokenizer, tokenize a sequence, and use it with a Hugging Face transformer model:

```python
from ape_tokenizer import APETokenizer
from transformers import BertModel

# Initialize the tokenizer and train it
tokenizer = APETokenizer()
corpus = ["CCO", "C=O", "CCC", "CCN"]
tokenizer.train(corpus, max_vocab_size=100)

# Tokenize a SMILES string
smiles = "CCO"
encoded = tokenizer(smiles, add_special_tokens=True, return_tensors="pt")
print("Encoded:", encoded)

# Load a pre-trained BERT model (for demonstration purposes)
model = BertModel.from_pretrained("bert-base-uncased")

# Use the tokenized input in the model
outputs = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])

# Model outputs
print(outputs)
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

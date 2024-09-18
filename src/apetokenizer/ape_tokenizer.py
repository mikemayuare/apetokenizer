from collections import defaultdict
import re
import json
import os


class APETokenizer:
    def __init__(
        self,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.vocabulary_frequency = defaultdict(int)
        self.pair_counts = defaultdict(int)
        self.special_tokens = {
            self.bos_token: 0,
            self.pad_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
            self.mask_token: 4,
        }
        self.vocabulary = dict(self.special_tokens)
        self.update_reverse_vocabulary()

    @property
    def bos_token_id(self):
        return self.special_tokens[self.bos_token]

    @property
    def eos_token_id(self):
        return self.special_tokens[self.eos_token]

    @property
    def pad_token_id(self):
        return self.special_tokens[self.pad_token]

    @property
    def mask_token_id(self):
        return self.special_tokens[self.mask_token]

    def __call__(
        self,
        text,
        padding=False,
        max_length=None,
        add_special_tokens=False,
        return_tensors=None,
    ):
        """
        Tokenize and prepare the input text.

        :param text: str, the text to tokenize and encode.
        :param add_special_tokens: bool, whether to add special tokens (like <s> and </s>).
        :param max_length: int, the maximum length of the token sequence.
        :param return_tensors: str, the type of tensors to return ('pt' for PyTorch, 'tf' for TensorFlow).
        :return: A dictionary with tokenized and encoded information.
        """
        # Encode the text using the encode method
        encoded_inputs = self.encode(
            text,
            padding=padding,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
        )

        # Create a dictionary to hold the output
        outputs = {"input_ids": encoded_inputs}

        # Calculate the attention mask (1 for tokens, 0 for padding)
        attention_mask = [
            1 if token_id != self.vocabulary[self.pad_token] else 0
            for token_id in encoded_inputs
        ]
        outputs["attention_mask"] = attention_mask

        # Truncate the sequences to max_length if necessary
        if max_length is not None:
            outputs["input_ids"] = outputs["input_ids"][:max_length]
            outputs["attention_mask"] = outputs["attention_mask"][:max_length]

        # Convert outputs to tensors if return_tensors is specified
        if return_tensors == "pt":  # For PyTorch
            import torch

            outputs["input_ids"] = torch.tensor(outputs["input_ids"])
            outputs["attention_mask"] = torch.tensor(outputs["attention_mask"])
        elif return_tensors == "tf":  # For TensorFlow
            import tensorflow as tf

            outputs["input_ids"] = tf.convert_to_tensor(outputs["input_ids"])
            outputs["attention_mask"] = tf.convert_to_tensor(outputs["attention_mask"])

        return outputs

    def __len__(self):
        """
        Return the number of tokens in the tokenizer's vocabulary.
        """
        return len(self.vocabulary)

    def pre_tokenize(self, molecule):
        """Pretokenize SMILES or SELFIES

        Args:
            molecule (str): SMILES or SELFIES strings
            type (str): Type of inputs, "smiles" for SMILES or "selfies" for SELFIES. Defaults to "selfies".

        Returns:
            list: Tokenized molecules.
        """
        # words = re.findall(r"\[[^\]]*\]", molecule)

        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        words = re.findall(pattern, molecule)

        return words

    def train(
        self,
        corpus,
        type="selfies",
        max_vocab_size: int = 5000,
        min_freq_for_merge: int = 2000,
        save_checkpoint: bool = False,
        checkpoint_path: str = "checkpoint",
        checkpoint_interval=500,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq_for_merge = min_freq_for_merge
        # self.max_token_length = max_token_length

        text_padding = " " * 80

        # Preprocessing: Tokenize and count word frequencies upfront
        print("Pretokenizing", end="\r")
        words = [word for sentence in corpus for word in self.pre_tokenize(sentence)]
        vocabulary_frequency = defaultdict(int)
        for word in words:
            vocabulary_frequency[word] += 1
        print(
            f"Pretokenization complete, found {len(vocabulary_frequency)} tokens",
            end="\r",
        )

        # to add the pretokens to the vocabulary numbering
        pre_tokens_counts = len(vocabulary_frequency)

        # Function optimized to reduce dictionary lookups
        def get_most_common_pair(words):
            for i in range(len(words) - 1):
                pair = (words[i], words[i + 1])
                self.pair_counts[pair] += 1

            # Minimize lookups by using max function directly
            most_common_pair, freq = max(
                self.pair_counts.items(), key=lambda x: x[1], default=((None, None), 0)
            )
            return most_common_pair, freq

        merged_counter = len(vocabulary_frequency) + 1
        checkpoint_increment = checkpoint_interval
        batch = checkpoint_interval + pre_tokens_counts

        while True:
            if save_checkpoint:
                if len(vocabulary_frequency) == batch:
                    self.vocabulary_frequency = dict(vocabulary_frequency)
                    self.vocabulary = {
                        **self.special_tokens,
                        **{
                            word: idx
                            for idx, word in enumerate(
                                vocabulary_frequency.keys(),
                                start=len(self.special_tokens),
                            )
                        },
                    }

                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)

                    self.save_vocabulary(f"{checkpoint_path}/checkpoint_{batch}.json")
                    print(
                        f"Checkpoint saved at {checkpoint_path}/checkpoint_{batch}.json"
                    )
                    self.save_pretrained(f"{checkpoint_path}/checkpoint_{batch}")
                    batch += checkpoint_increment

            if len(vocabulary_frequency) > self.max_vocab_size:
                print("\rMax vocabulary achieved", text_padding)
                break

            most_common_pair, freq = get_most_common_pair(words)
            if freq < self.min_freq_for_merge:
                print("\rNot enough frequency found", text_padding)
                break

            merged_word = "".join(most_common_pair)
            if merged_word not in vocabulary_frequency.keys():
                print(
                    f"New merge found: {merged_word} {merged_counter}/{max_vocab_size} {round(merged_counter / max_vocab_size * 100, 2)}%"
                )
                merged_counter += 1
            merged_word_freq = vocabulary_frequency.get(merged_word, 0)
            vocabulary_frequency[merged_word] = merged_word_freq + freq

            # Minimize dictionary lookups inside the loop
            new_words = []
            skip_next = False
            for i in range(len(words)):
                if skip_next:
                    skip_next = False
                    continue

                # Look ahead to minimize lookups
                if (
                    i < len(words) - 1
                    and words[i] == most_common_pair[0]
                    and words[i + 1] == most_common_pair[1]
                ):
                    new_words.append(merged_word)
                    skip_next = True
                else:
                    new_words.append(words[i])

            words = new_words

        # Convert vocabulary_frequency to a regular dictionary for final output
        self.vocabulary_frequency = dict(vocabulary_frequency)
        self.vocabulary = {
            **self.special_tokens,
            **{
                word: idx
                for idx, word in enumerate(vocabulary_frequency.keys(), start=5)
            },
        }
        print("\nTraining complete.")

        return None

    def pad(
        self,
        batch,
        padding=False,
        return_tensors=None,
        pad_to_multiple_of=None,
        **kwargs,
    ):
        # Determine the maximum length in this batch for padding
        max_length = max(len(seq["input_ids"]) for seq in batch)

        if pad_to_multiple_of:
            # Ensure max_length is a multiple of pad_to_multiple_of
            max_length = (
                (max_length - 1) // pad_to_multiple_of + 1
            ) * pad_to_multiple_of

        padded_sequences = []
        attention_masks = []
        labels = []  # Prepare to collect labels
        for seq in batch:
            # Extract the input_ids from the current sequence (assuming it's a dictionary)
            input_ids = seq["input_ids"]
            padding_length = max_length - len(input_ids)

            # Create the padded sequence and attention mask
            padded_seq = input_ids + [self.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids) + [0] * padding_length

            padded_sequences.append(padded_seq)
            attention_masks.append(attention_mask)

            # Handle labels if they are present in the batch
            if "labels" in seq:
                labels.append(seq["labels"])

        # Convert to tensors or the appropriate format
        if return_tensors == "pt":
            import torch

            padded_sequences = torch.tensor(padded_sequences)
            attention_masks = torch.tensor(attention_masks)
            output = {"input_ids": padded_sequences, "attention_mask": attention_masks}
            if labels:
                output["labels"] = torch.tensor(labels)
            return output
        elif return_tensors == "tf":  # For TensorFlow
            import tensorflow as tf

            padded_sequences = tf.convert_to_tensor(padded_sequences)
            attention_masks = tf.convert_to_tensor(attention_masks)
            output = {"input_ids": padded_sequences, "attention_mask": attention_masks}
            if labels:
                output["labels"] = tf.convert_to_tensor(labels)
            return output
        else:
            # Return as lists if tensors are not requested
            output = {"input_ids": padded_sequences, "attention_mask": attention_masks}
            if labels:
                output["labels"] = labels
            return output

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        """
        Retrieves a mask array indicating which tokens are special tokens.

        :param token_ids: List[int], the tokenized code of the text.
        :param already_has_special_tokens: bool, whether the token_ids already contain special tokens.
        :return: List[int], a list of the same length as token_ids, where 1 indicates a special token.
        """
        if already_has_special_tokens:
            return [
                1 if token in self.special_tokens.values() else 0 for token in token_ids
            ]

        # If the sequence doesn't have special tokens, we need to add them.
        # This example assumes that the beginning and end of the sequence are special.
        return [1] + ([0] * (len(token_ids) - 2)) + [1]

    def train_from_iterator(self, iterator):
        pass

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):  # Single token
            return self.vocabulary.get(tokens, self.vocabulary[self.unk_token])
        else:  # List of tokens
            return [
                self.vocabulary.get(token, self.vocabulary[self.unk_token])
                for token in tokens
            ]
            
    def update_reverse_vocabulary(self):
        """Updates the reverse vocabulary based on the current state of the vocabulary."""
        # Create a reverse mapping from IDs to tokens
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts a sequence of token IDs back to a list of string tokens.

        :param token_ids: List[int], a list of token IDs.
        :return: List[str], a list of string tokens corresponding to the token IDs.
        """
        # Map each token ID to its corresponding string token
        return [self.reverse_vocabulary.get(token_id, self.unk_token) for token_id in token_ids]

    def encode(self, text, padding=False, max_length=None, add_special_tokens=False):
        """
        Encodes a given text into a sequence of vocabulary indices.

        :param text: String, the text to encode.
        :param add_special_tokens: Boolean, whether to add bos and eos tokens.
        :param max_length: Int, the maximum length of the sequence with padding.
        :param padding: Boolean or String, False for no padding, True for default padding, or "max_length" for max_length padding.
        :return: List of integers, the encoded text.
        """
        # Initialize the list of encoded tokens
        encoded_tokens = []

        # Add the Beginning of String token
        if add_special_tokens:
            encoded_tokens.append(self.vocabulary[self.bos_token])

        # Scan and tokenize the text based on the vocabulary
        i = 0
        while i < len(text):
            match = None
            # Check for the longest sequence in the vocabulary that matches the text
            for j in range(len(text), i, -1):
                possible_match = text[i:j]
                if possible_match in self.vocabulary:
                    match = possible_match
                    break
            if match:
                # Add the token's index to the encoded tokens
                encoded_tokens.append(self.vocabulary[match])
                i += len(match)  # Move past the matched text
            else:
                # If no match is found, use the unknown token and move one character forward
                encoded_tokens.append(self.vocabulary[self.unk_token])
                i += 1

        if add_special_tokens:
            encoded_tokens.append(self.vocabulary[self.eos_token])

        # Handle padding if required
        if padding:
            pad_token = self.vocabulary[
                self.pad_token
            ]  # Assuming you have a PAD token in your vocabulary
            if max_length is None:
                raise ValueError(
                    "max_length must be specified if padding is True or 'max_length'"
                )

            # Add padding tokens until the sequence is of max_length
            while len(encoded_tokens) < max_length:
                encoded_tokens.append(pad_token)

        return encoded_tokens

    def save_vocabulary(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                self.vocabulary,
                f,
                ensure_ascii=False,
                indent=4,
            )
        with open(f"{file_path.rstrip('.json')}_freq.json", "w", encoding="utf-8") as f:
            json.dump(
                self.vocabulary_frequency,
                f,
                ensure_ascii=False,
                indent=4,
            )

    def load_vocabulary(self, file_path):
        with open(file_path, "r", encoding="utf_8") as f:
            self.vocabulary = json.load(f)
        
        self.update_reverse_vocabulary()
        # with open(f"{file_path.rstrip('.json')}_freq.json", "r", encoding="utf_8") as f:
        #     self.vocabulary_frequency = json.load(f)

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=4)

        # Save special tokens
        special_tokens_file = os.path.join(save_directory, "special_tokens.json")
        with open(special_tokens_file, "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=4)

        # Save training state
        # Prepare the data to be JSON serializable
        vocabulary_frequency_serializable = {str(k): v for k, v in self.vocabulary_frequency.items()}
        pair_counts_serializable = {str(k): v for k, v in self.pair_counts.items()}
        
        training_state = {
            "vocabulary_frequency": vocabulary_frequency_serializable,
            "pair_counts": pair_counts_serializable,
        }

        training_state_file = os.path.join(save_directory, "training_state.json")
        with open(training_state_file, "w", encoding="utf-8") as f:
            json.dump(training_state, f, ensure_ascii=False, indent=4)
        
        print(f"Tokenizer and training state saved in {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_directory):
        vocab_file = os.path.join(pretrained_directory, "vocab.json")
        special_tokens_file = os.path.join(pretrained_directory, "special_tokens.json")
        training_state_file = os.path.join(pretrained_directory, "training_state.json")

        # Load vocabulary
        if os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocabulary = json.load(f)
        else:
            raise FileNotFoundError(f"Vocabulary file {vocab_file} not found.")

        # Load special tokens
        if os.path.isfile(special_tokens_file):
            with open(special_tokens_file, "r", encoding="utf-8") as f:
                special_tokens = json.load(f)
        else:
            raise FileNotFoundError(
                f"Special tokens file {special_tokens_file} not found."
            )

        # Initialize the tokenizer
        tokenizer = cls()
        tokenizer.vocabulary = vocabulary
        tokenizer.special_tokens = special_tokens

        # Load training state if it exists
        if os.path.isfile(training_state_file):
            with open(training_state_file, "r", encoding="utf-8") as f:
                training_state = json.load(f)
            tokenizer.vocabulary_frequency = defaultdict(
                int, training_state["vocabulary_frequency"]
            )
            tokenizer.pair_counts = defaultdict(int, training_state["pair_counts"])

        return tokenizer

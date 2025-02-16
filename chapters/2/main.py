import os
import re
import tiktoken
import torch
from importlib.metadata import version
from dataset import GPTDatasetV1
from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2
from torch.utils.data import Dataset, DataLoader


# Regex tokenizer
def regTokenizer():
    # Read file
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Tokenize
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    # Create vocabulary
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}

    print(len(vocab.items()))
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    # Test tokenizer1
    # tokenizer = SimpleTokenizerV1(vocab)
    # text = """"It's the last he painted, you know,"
    #        Mrs. Gisburn said with pardonable pride."""
    # ids = tokenizer.encode(text)
    # print(ids)
    # print(tokenizer.decode(ids))

    # Test tokenizer2
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    tokenizer2 = SimpleTokenizerV2(vocab)
    print(tokenizer2.encode(text))
    print(tokenizer2.decode(tokenizer2.encode(text)))


# BPE tokenizer
def bpeTokenizer():
    tokenizer = tiktoken.get_encoding('gpt2')
    text = (
        'Hello, do you like tea? <|endoftext|> In the sunlit terraces50'
        'of someunknownPlace.'
    )
    integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    print(integers)
    strings = tokenizer.decode(integers)
    print(strings)


# Exercise 2.1
def ex2_1():
    tokenizer = tiktoken.get_encoding('gpt2')
    text = 'Akwirw ier'
    tokens = tokenizer.encode(text)
    print('Tokens:', tokens)
    strings = tokenizer.decode(tokens)
    print('Decoded:', strings)


# Playground
def playground():
    tokenizer = tiktoken.get_encoding('gpt2')
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # BPE
    enc_text = tokenizer.encode(raw_text)
    print('Total tokens:', len(enc_text))
    print()
    
    # Remove first 50 tokens
    enc_sample = enc_text[50:]

    print('Input/Target; shift by one')
    context_size = 4                 # determines how many tokens are included in input
    x = enc_sample[:context_size]    # input
    y = enc_sample[1:context_size+1] # target
    print(f'x: {x}')
    print(f'y:      {y}')


# Sample data w/ sliding window
def dataloader():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    tokenizer = tiktoken.get_encoding('gpt2')
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    # Remove first 50 tokens from sample
    enc_sample = enc_text[50:]

    # Defines number of tokens included in input
    context_size = 4

    # x: input tokens, y: targets
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f'x: {x}')
    print(f'y:      {y}')

    # Create next-word prediction tasks
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        #print(context, '---->', desired)
        print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))


# Dataloader to generate batches
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# Test create_dataloader_v1
def test_create_dataloader_v1():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)


# Exercise 2.2
def ex2_2():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=2, stride=2, shuffle=False
    )
    data_iter = iter(dataloader)
    print('Max Length: 2, Stride: 2')
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=8, stride=2, shuffle=False
    )
    data_iter = iter(dataloader)
    print('Max Length: 8, Stride: 2')
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)


# Test create_dataloader_v1 with greater batch sizes
def test_batch_create_dataloader_v1():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    print('Batch Size: 8, Max Length: 4, Stride: 4')
    inputs, targets = next(data_iter)
    print('Inputs:\n', inputs)
    print('Targets:\n', targets)


# Experiment w/ generating embeddings for token IDs
def embed():
    torch.manual_seed(123)
    vocab_size = 6
    output_dim = 3
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)


# Embedding layer w/ absolute positional encoding
def abs_positional_encoding():
    # Create embedding matrix
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # Instantiate dataloader
    script_dir = os.path.dirname(__file__)
    rel_path = '../the-verdict.txt'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, 
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print('Token IDs:\n', inputs)
    print('Inputs shape:\n', inputs.shape)

    # Embed the input tokens
    token_embeddings = token_embedding_layer(inputs)
    print('\nToken Embeddings shape:\n', token_embeddings.shape)

    # Create positional embedding layer
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print('\nPos Embedding Tensor shape:\n', pos_embeddings.shape)

    # Add pos embedding tensor to each token embedding tensor to get input embeddings
    input_embeddings = token_embeddings + pos_embeddings
    print('\nInput Embeddings shape:\n', input_embeddings.shape)


if __name__ == '__main__':
    abs_positional_encoding()
import tensorflow as tf
import os,sys
import numpy as np

def test_demo(sess, trainable_model, generated_num, batch_size, output_file, vocab, iter):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    with open(output_file, 'a') as fout:
        fout.write("==================  iter: {}  ==================\n".format(iter))
        print("==================  iter: {}  ==================\n".format(iter))
        for poem in generated_samples:
            buffer = ' '.join([vocab.get(x, '_') for x in poem]) + '\n'
            fout.write(buffer)
            print(buffer.strip())

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def load_vocab(vacab_file):
    vocab_w2idx = {}
    vocab_idx2w = {}

    with open(vacab_file) as f:
        for line in f:
            line = line.strip()
            idx, w, feq = line.split('\\')
            idx = int(idx)
            vocab_idx2w[idx] = w
            vocab_w2idx[w] = idx
    return vocab_w2idx, vocab_idx2w, len(vocab_w2idx)
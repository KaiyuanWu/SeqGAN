# test whether the data prepration is correct
import os, sys
sys.path.insert(0, os.path.abspath('..'))
import utils
import dataloader



if __name__ == "__main__":
    vocab_file = '../script/work/char_vocab_feq_gt1.txt'
    data_file = '../script/work/char_doc_idx_feq_gt1.txt'
    vocab_w2idx, vocab_idx2w, len_vocab_w2idx = utils.load_vocab(vocab_file)
    train_data = dataloader.Gen_Data_loader(batch_size = 32, seq_len=42)
    train_data.create_batches(data_file)

    num_test_batches = 1
    for idx in range(num_test_batches):
        batch = train_data.next_batch()
        for s in batch:
            print s
            poem = ''.join([vocab_idx2w.get(i,'0') for i in s])
            print poem



import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import utils
import cPickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 64 # embedding dimension
HIDDEN_DIM = 64 # hidden state dimension of lstm cell
SEQ_LENGTH = 42 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 100 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 20000
positive_file = 'script/work/doc_idx_feq_gt1.txt'
negative_file = 'script/work/generator_sample.txt'
eval_file = 'script/work/eval_file.txt'
vacab_file = 'script/work/vocab_feq_gt1.txt'
generated_num = 10000

vocab_size = 9631

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0
    vocab_w2idx, vocab_idx2w, len_vocab_w2idx = utils.load_vocab(vacab_file)

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, seq_len=SEQ_LENGTH)
    dis_data_loader = Dis_dataloader(BATCH_SIZE, seq_len=SEQ_LENGTH)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    gen_data_loader.create_batches(positive_file)
    print("gen_data_loader num_batches: ", gen_data_loader.num_batch)


    #  pre-train generator
    print 'Start pre-training...'
    for epoch in xrange(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 1 == 0:
            utils.test_demo(sess, generator, 8, 8, eval_file, vocab_idx2w, epoch)
            print 'pre-train epoch ', epoch, 'train_loss ', loss

    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(10):
        utils.generate_samples(sess, generator,  64, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            print 'adversarial-train epoch ', total_batch
            utils.test_demo(sess, generator, 8, 8, eval_file, vocab_idx2w, total_batch)
        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle
import getopt
import sys
import os

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
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
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
test_file = 'save/test_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000

save_every = 5
default_checkpoint_folder = "./checkpoints"

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def eval_model(generator, target, test_data_loader, sess):
    #import pdb; pdb.set_trace()
    gen_probs = []
    target_probs = []
    for it in xrange(test_data_loader.num_batch):
        batch = test_data_loader.next_batch()
        gen_prob = generator.get_logprobs(sess,batch)
        gen_probs.extend(gen_prob)

        target_prob = target.get_logprobs(sess,batch)
        target_probs.extend(target_prob)


    diff = 0.
    for i in range(len(target_probs)):
        diff += abs(target_probs[i] - gen_probs[i])
    #import pdb; pdb.set_trace()
    return diff / float(len(target_probs))




def main():
    optlist, args = getopt.gnu_getopt(sys.argv, 'd:')
    opts = dict(optlist)
    checkpoint_folder = opts['-d'] if '-d' in opts else default_checkpoint_folder

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    global_step =tf.Variable(tf.zeros([1], dtype=tf.int32), name="global_step")
    local_step =tf.Variable(tf.zeros([1], dtype=tf.int32), name="local_step")
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    increment_local_step_op = tf.assign(local_step, local_step+1)
    reset_local_op = tf.assign(local_step, [0])

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    test_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 5000
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = cPickle.load(open('save/target_params.pkl'))
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    log = open('save/experiment-log.txt', 'w')

    if os.path.isfile(checkpoint_folder + "/checkpoint"):
        saver.restore(sess, checkpoint_folder + "/model.ckpt")
        #TODO see if we can reduce this down
        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        #generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    else:
        sess.run(tf.global_variables_initializer())


    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)



    #Generates test data
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, test_file)
    #TODO check necessity
    test_data_loader.create_batches(test_file)




    if sess.run(global_step) == 0:
        #  pre-train generator
        print 'Start pre-training...'
        log.write('pre-training...\n')
        for epoch in xrange(PRE_EPOCH_NUM - sess.run(local_step)):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            print "Our loss metric:"
            print eval_model(generator, target_lstm, test_data_loader, sess)
            if epoch % 5 == 0:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print 'pre-train epoch ', epoch, 'test_loss ', test_loss
                buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                log.write(buffer)
            sess.run(increment_local_step_op)
        sess.run(increment_global_step_op)
        sess.run(reset_local_op)

    if sess.run(global_step) == 1:
        print 'Start pre-training discriminator...'
        # Train 3 epoch on the generated data and do this for 50 times
        for i in range(50 - sess.run(local_step)):
            print 'Round %d of discriminator pre-training' % i
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
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
                    sess.run(increment_local_step_op)
        sess.run(increment_global_step_op)
        sess.run(reset_local_op)
    if sess.run(global_step) == 2:
        rollout = ROLLOUT(generator, 0.)

        print '#########################################################################'
        print 'Start Adversarial Training...'
        log.write('adversarial training...\n')
        for total_batch in range(TOTAL_BATCH - sess.run(local_step)):
            if total_batch % save_every == 0:
                saver.save(sess, default_checkpoint_folder + "/model.ckpt")
            # Train the generator for one step
            for it in range(1):
                samples = generator.generate(sess)
                rewards = rollout.get_reward(sess, samples, 16, discriminator)
                feed = {generator.x: samples, generator.rewards: rewards}
                _ = sess.run(generator.g_updates, feed_dict=feed)

            # Test
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                likelihood_data_loader.create_batches(eval_file)
                test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                print 'total_batch: ', total_batch, 'test_loss: ', test_loss
                log.write(buffer)

            # Update roll-out parameters
            rollout.update_params()

            # Train the discriminator
            for _ in range(5):
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
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
            sess.run(increment_local_step_op)

    log.close()


if __name__ == '__main__':
    main()

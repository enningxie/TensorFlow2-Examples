# coding=utf-8
'''
Trains two recurrent neural networks based upon a story and a question.
'''
import tarfile
import re
from functools import reduce
import numpy as np
import tensorflow as tf
from argparse import Namespace

# 指定GPU并配置GPU
PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if PHYSICAL_DEVICES:
    USED_GPUS = PHYSICAL_DEVICES[2:3]
    tf.config.experimental.set_visible_devices(devices=USED_GPUS, device_type='GPU')
    for tmp_gpu in USED_GPUS:
        tf.config.experimental.set_memory_growth(device=tmp_gpu, enable=True)

ARGS = Namespace(
    # Training parameters.
    embed_hidden_size=50,
    sent_hidden_size=100,
    query_hidden_size=100,
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    model_path='./babi_rnn.h5',
)


class DataLoader(object):
    def __init__(self):
        self.train_data, self.val_data = self.gen_data()

    def gen_data(self):
        # Download data
        path = tf.keras.utils.get_file('babi-tasks-v1-2.tar.gz',
                                       origin='https://s3.amazonaws.com/text-datasets/'
                                              'babi_tasks_1-20_v1-2.tar.gz')

        # Default QA1 with 1000 samples
        # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
        # QA1 with 10,000 samples
        # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
        # QA2 with 1000 samples
        challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
        # QA2 with 10,000 samples
        # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
        with tarfile.open(path) as tar:
            train = self.get_stories(tar.extractfile(challenge.format('train')))
            test = self.get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train + test:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        self.vocab_size = len(vocab) + 1
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        self.story_maxlen = max(map(len, (x for x, _, _ in train + test)))
        self.query_maxlen = max(map(len, (x for _, x, _ in train + test)))

        x, xq, y = self.vectorize_stories(train, word_idx, self.story_maxlen, self.query_maxlen)
        tx, txq, ty = self.vectorize_stories(test, word_idx, self.story_maxlen, self.query_maxlen)
        return (x, xq, y), (tx, txq, ty)

    def tokenize(self, sent):
        '''Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''
        return [x.strip() for x in re.split(r'(\W+)', sent) if x.strip()]

    def parse_stories(self, lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true,
        only the sentences that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, only_supporting=False, max_length=None):
        '''Given a file name, read the file, retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data
                if not max_length or len(flatten(story)) < max_length]
        return data

    def vectorize_stories(self, data, word_idx, story_maxlen, query_maxlen):
        xs = []
        xqs = []
        ys = []
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            y = np.zeros(len(word_idx) + 1)
            y[word_idx[answer]] = 1
            xs.append(x)
            xqs.append(xq)
            ys.append(y)
        return (tf.keras.preprocessing.sequence.pad_sequences(xs, maxlen=story_maxlen),
                tf.keras.preprocessing.sequence.pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))


class BabiRNN(object):
    def __init__(self):
        self.data_loader = DataLoader()
        self.model = self.create_model()

    def create_model(self):
        # sentence input
        inputs_0 = tf.keras.layers.Input(shape=(self.data_loader.story_maxlen,), dtype='int32')
        x_0 = tf.keras.layers.Embedding(self.data_loader.vocab_size, ARGS.embed_hidden_size)(inputs_0)
        x_0 = tf.keras.layers.LSTM(ARGS.sent_hidden_size)(x_0)

        # question input
        inputs_1 = tf.keras.layers.Input(shape=(self.data_loader.query_maxlen,), dtype='int32')
        x_1 = tf.keras.layers.Embedding(self.data_loader.vocab_size, ARGS.embed_hidden_size)(inputs_1)
        x_1 = tf.keras.layers.LSTM(ARGS.query_hidden_size)(x_1)

        x = tf.keras.layers.concatenate([x_0, x_1])
        outputs = tf.keras.layers.Dense(self.data_loader.vocab_size, activation='softmax')(x)

        return tf.keras.models.Model([inputs_0, inputs_1], outputs)

    def get_dataset(self, data, is_training=False, return_steps=False):
        x_data, xq_data, y_data = data
        tmp_dataset = tf.data.Dataset.from_tensor_slices(((x_data, xq_data), y_data))
        if is_training:
            tmp_dataset = tmp_dataset.shuffle(buffer_size=1024).batch(ARGS.batch_size)
        else:
            tmp_dataset = tmp_dataset.batch(ARGS.batch_size)
        if return_steps:
            if x_data.shape[0] % ARGS.batch_size == 0:
                tmp_steps = x_data.shape[0] // ARGS.batch_size
            else:
                tmp_steps = x_data.shape[0] // ARGS.batch_size + 1
            return tmp_dataset, tmp_steps
        else:
            return tmp_dataset

    # custom training loop
    def train(self):
        # instantiate an optimizer to train the model.
        optimizer = tf.keras.optimizers.Adam(learning_rate=ARGS.learning_rate)
        # instantiate a loss function.
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # prepare the metrics.
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        # prepare the training dataset.
        train_dataset, train_steps = self.get_dataset(self.data_loader.train_data, is_training=True,
                                                      return_steps=True)

        # Prepare the validation dataset.
        val_dataset, val_steps = self.get_dataset(self.data_loader.val_data, return_steps=True)

        # Iterate over epochs.
        best_val_acc = 0.
        for epoch in range(ARGS.epochs):
            print('*********************')
            print('Epoch {} training...'.format(epoch))
            training_bar = tf.keras.utils.Progbar(train_steps, stateful_metrics=['loss', 'acc'])

            # Iterate over the batches of the dataset.
            for train_step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Update training metric.
                train_acc_metric(y_batch_train, logits)

                # Logging
                training_bar.update(train_step + 1,
                                    values=[('loss', float(loss_value)), ('acc', float(train_acc_metric.result()))])

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            validating_bar = tf.keras.utils.Progbar(val_steps, stateful_metrics=['val_acc'])
            # Run a validation loop at the end of each epoch.
            for val_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                val_logits = self.model(x_batch_val)
                # Update val metrics
                val_acc_metric(y_batch_val, val_logits)
                # Logging
                validating_bar.update(val_step + 1, values=[('val_acc', float(val_acc_metric.result()))])
            val_acc = val_acc_metric.result()
            # Save the best model with the highest verification accuracy
            if val_acc > best_val_acc:
                print('model saving...')
                # todo tf.saved_model.save
                # normal
                self.model.save_weights(ARGS.model_path)
                # # new
                # tf.saved_model.save(self.model, args.model_dir_path)
                best_val_acc = val_acc
            val_acc_metric.reset_states()


if __name__ == '__main__':
    tmp_model = BabiRNN()
    tmp_model.train()

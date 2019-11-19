# coding=utf-8
'''
# An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
'''

import tensorflow as tf
import numpy as np
from argparse import Namespace

# 指定GPU并配置GPU
PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if PHYSICAL_DEVICES:
    USED_GPUS = PHYSICAL_DEVICES[:1]
    tf.config.experimental.set_visible_devices(devices=USED_GPUS, device_type='GPU')
    for tmp_gpu in USED_GPUS:
        tf.config.experimental.set_memory_growth(device=tmp_gpu, enable=True)

ARGS = Namespace(
    # colors
    ok='\033[92m',
    fail='\033[91m',
    close='\033[0m',
    # Parameters for the model and dataset.
    chars='0123456789+ ',
    training_size=50000,
    digits=3,
    reverse=True,
    # Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
    # int is DIGITS.
    maxlen=7,
    # Training parameters.
    hidden_size=128,
    batch_size=128,
    layers=1,
    epochs=200,
    learning_rate=0.001,
    model_path='./addition_rnn.h5',
    model_dir_path='/Data/xen/Codes/SoucheLab/TensorFlow2-Examples/examples/saved_models'
)


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class DataLoader(object):
    def __init__(self):
        # All the numbers, plus sign and space for padding.
        self.ctable = CharacterTable(ARGS.chars)
        self.questions, self.expected = self.gen_data()
        self.train_data, self.val_data = self.vectorization()

    def gen_data(self):
        questions = []
        expected = []
        seen = set()
        print('Generating data...')
        while len(questions) < ARGS.training_size:
            f = lambda: int(''.join(np.random.choice(list('0123456789'))
                                    for i in range(np.random.randint(1, ARGS.digits + 1))))
            a, b = f(), f()
            # Skip any addition questions we've already seen
            # Also skip any such that x+Y == Y+x (hence the sorting).
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            # Pad the data with spaces such that it is always MAXLEN.
            q = '{}+{}'.format(a, b)
            query = q + ' ' * (ARGS.maxlen - len(q))
            ans = str(a + b)
            # Answers can be of maximum size DIGITS + 1.
            ans += ' ' * (ARGS.digits + 1 - len(ans))
            if ARGS.reverse:
                # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
                # space used for padding.)
                query = query[::-1]
            questions.append(query)
            expected.append(ans)
        print('Total addition questions:', len(questions))
        return questions, expected

    def vectorization(self):
        print('Vectorization...')
        x = np.zeros((len(self.questions), ARGS.maxlen, len(ARGS.chars)), dtype=np.float)
        y = np.zeros((len(self.questions), ARGS.digits + 1, len(ARGS.chars)), dtype=np.float)
        for i, sentence in enumerate(self.questions):
            x[i] = self.ctable.encode(sentence, ARGS.maxlen)
        for i, sentence in enumerate(self.expected):
            y[i] = self.ctable.encode(sentence, ARGS.digits + 1)
        # Shuffle (x, y) in unison as the later parts of x will almost all be larger
        # digits.
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        # Explicitly set apart 10% for validation data that we never train over.
        split_at = len(x) - len(x) // 10
        (x_train, x_val) = x[:split_at], x[split_at:]
        (y_train, y_val) = y[:split_at], y[split_at:]
        print('Training Data:')
        print(x_train.shape)
        print(y_train.shape)

        print('Validation Data:')
        print(x_val.shape)
        print(y_val.shape)

        return (x_train, y_train), (x_val, y_val)


class AdditionRNN(object):
    def __init__(self):
        self.model = self.create_model()
        self.data_loader = DataLoader()

    def create_model(self):
        inputs = tf.keras.layers.Input(shape=(ARGS.maxlen, len(ARGS.chars)))
        x = tf.keras.layers.LSTM(ARGS.hidden_size)(inputs)
        x = tf.keras.layers.RepeatVector(ARGS.digits + 1)(x)
        for _ in range(ARGS.layers):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            x = tf.keras.layers.LSTM(ARGS.hidden_size, return_sequences=True)(x)
        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(ARGS.chars), activation='softmax'))(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def get_dataset(self, data, is_training=False, return_steps=False):
        x_data, y_data = data
        tmp_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
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
        train_dataset, train_steps = self.get_dataset(self.data_loader.train_data, is_training=True, return_steps=True)

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
                # tf.saved_model.save(self.model, ARGS.model_dir_path)
                best_val_acc = val_acc
            val_acc_metric.reset_states()

    def evaluate(self):
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        self.model.load_weights(ARGS.model_path)
        x_val, y_val = self.data_loader.val_data
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = self.model.predict(rowx, verbose=0)
            # predict_classes
            if preds.shape[-1] > 1:
                preds = preds.argmax(axis=-1)
            else:
                preds = (preds > 0.5).astype('int32')
            q = self.data_loader.ctable.decode(rowx[0])
            correct = self.data_loader.ctable.decode(rowy[0])
            guess = self.data_loader.ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if ARGS.reverse else q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(ARGS.ok + '☑' + ARGS.close, end=' ')
            else:
                print(ARGS.fail + '☒' + ARGS.close, end=' ')
            print(guess)


if __name__ == '__main__':
    tmp_model = AdditionRNN()
    tmp_model.train()

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

from tensorflow import name_scope

from functools import partial

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn.ops import gen_gru_ops
from tensorflow.python.ops import init_ops

class DynamicMemoryCell(tf.contrib.rnn.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.
    """

    def __init__(self,
                 num_blocks,
                 num_units_per_block,
                 keys,
                 initializer=None,
                 recurrent_initializer=None,
                 activation=tf.nn.relu,):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._keys = keys
        self._activation = activation # \phi
        self._initializer = initializer
        self._recurrent_initializer = recurrent_initializer

    @property
    def state_size(self):
        "Return the total state size of the cell, across all blocks."
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        "Return the total output size of the cell, across all blocks."
        return self._num_blocks

    def zero_state(self, batch_size, dtype):
        "Initialize the memory to the key values."
        zero_state = tf.concat([tf.expand_dims(key, axis=0) for key in self._keys], axis=1)
        zero_state_batch = tf.tile(zero_state, [batch_size, 1])
        return zero_state_batch

    def get_gate(self, state_j, key_j, inputs, v=None, prev_a=None):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = tf.reduce_sum(inputs * state_j, axis=1)
        b = tf.reduce_sum(inputs * key_j, axis=1)
        return a + b

    def get_candidate(self, state_j, key_j, inputs, U, V, W, U_bias):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_V = tf.matmul(key_j, V)
        state_U = tf.matmul(state_j, U) + U_bias
        inputs_W = tf.matmul(inputs, W)
        return self._activation(state_U + inputs_W + key_V)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            U = tf.get_variable('U', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)
            V = tf.get_variable('V', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)
            W = tf.get_variable('W', [self._num_units_per_block, self._num_units_per_block],
                                initializer=self._recurrent_initializer)

            U_bias = tf.get_variable('U_bias', [self._num_units_per_block])

            state = tf.split(state, self._num_blocks, axis=1)
            assert len(state) == self._num_blocks

            gates = []
            next_states = []
            for j, state_j in enumerate(state): # Hidden State (j)
                key_j = tf.expand_dims(self._keys[j], axis=0)
                gate_j = self.get_gate(state_j, key_j, inputs)
                gates.append(tf.expand_dims(gate_j, -1))
                gate_j = tf.sigmoid(gate_j)
                candidate_j = self.get_candidate(state_j, key_j, inputs, U, V, W, U_bias)

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # Equation 5: h_j <- h_j / \norm{h_j}
                # Forget previous memories by normalization.
                state_j_next_norm = tf.norm(
                    tensor=state_j_next,
                    ord='euclidean',
                    axis=-1,
                    keep_dims=True)
                state_j_next_norm = tf.where(
                    tf.greater(state_j_next_norm, 0.0),
                    state_j_next_norm,
                    tf.ones_like(state_j_next_norm))
                state_j_next = state_j_next / state_j_next_norm

                next_states.append(state_j_next)
            gate_output = tf.concat(gates, axis=1)
            state_next = tf.concat(next_states, axis=1)
            return gate_output, state_next

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with name_scope(values=[t], name=name, default_name="zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(
            axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name
        )

def prelu(features, alpha, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU'):
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg


class EntNet_ROCStories(object):
    """End-To-End Memory Network."""
    def __init__(self, 
        batch_size, vocab_size, story_size, ending_size, embedding_size,
        embedding_mat=None,
        update_embeddings=False,
        max_grad_norm=5.0,
        n_keys=4,
        l2_final_layer=0.0,
        tagging_weight=0.5,
        initializer=tf.contrib.layers.xavier_initializer(),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        global_step=None,
        session=None,
        name='EntNet_ROCStories'):

        print name

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._story_size = story_size
        self._ending_size = ending_size
        self._embedding_size = embedding_size
        self._max_grad_norm = max_grad_norm
        self._init = initializer
        self._opt = optimizer
        self._global_step = global_step
        self._name = name
        self._embedding_mat = embedding_mat
        self._update_embeddings = update_embeddings
        self._n_keys = n_keys
        self._l2_final_layer = l2_final_layer
        self._tagging_weight = tagging_weight

        self._build_inputs()
        self._build_vars()
        logits, gate_output, stories_len = self._inference(
            self._stories, 
            self._endings,
            self._input_keep_prob,
            self._output_keep_prob,
            self._state_keep_prob,
            self._entnet_input_keep_prob,
            self._entnet_output_keep_prob,
            self._entnet_state_keep_prob,
            self._final_layer_keep_prob,
        )
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(self._answers, tf.float32), 
            name="cross_entropy"
        )
        # [None, 1]
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name="cross_entropy_mean"
        )

        self._gate_output = gate_output

        # gate supervision
        gate_output_pred = gate_output[:, :, :3]
        # [None, story_size, 3]

        gate_output_gold = self._stories_feat
        gate_output_gold = tf.cast(
            gate_output_gold, tf.float32
        )
        # [None, story_size, 3]]

        aux_loss = self._sequence_loss(gate_output_pred, gate_output_gold, stories_len)
        aux_loss = aux_loss * self._tagging_weight
        
        # l2 regularization
        trainable_variables = tf.trainable_variables()
        l2_loss_final_layer = 0.0
        assert self._l2_final_layer >= 0
        final_layer_weights = [ tf.nn.l2_loss(v) for v in trainable_variables
                                if 'R:0' in v.name]
        assert len(final_layer_weights) == 1
        l2_loss_final_layer = self._l2_final_layer * tf.add_n(final_layer_weights)

        loss_op = cross_entropy_mean + aux_loss + l2_loss_final_layer
        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)

        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, global_step=self._global_step, name="train_op")

        predict_op = logits
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op, feed_dict={self._input_embedding: self._embedding_mat})

    def _build_inputs(self):
        self._stories = tf.placeholder(
            tf.int32, [None, self._story_size], 
            name="stories"
        )
        self._endings = tf.placeholder(
            tf.int32, [None, self._ending_size], 
            name="endings"
        )
        self._answers = tf.placeholder(
            tf.int32, [None, 1], 
            name="answers"
        )
        self._stories_feat = tf.placeholder(
            tf.int32, [None, self._story_size, 3],
            name="stories_feat"
        )
        self._queries_feat = tf.placeholder(
            tf.int32, [None, self._ending_size, 3],
            name="queries_feat"
        )
        self._input_embedding = tf.placeholder(
            tf.float32, shape=self._embedding_mat.shape,
            name="input_embedding"
        )
        self._input_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="input_keep_prob"
        )
        self._output_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="output_keep_prob"
        )
        self._state_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="state_keep_prob"
        )
        self._entnet_input_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="entnet_input_keep_prob"
        )
        self._entnet_output_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="entnet_output_keep_prob"
        )
        self._entnet_state_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="entnet_state_keep_prob"
        )
        self._final_layer_keep_prob = tf.placeholder(
            tf.float32, shape=[],
            name="final_layer_keep_prob"
        )

    def _build_vars(self):
        with tf.variable_scope(self._name):
            self._embedding = tf.get_variable(
                name="embedding",
                dtype=tf.float32,
                initializer=self._input_embedding,
                trainable=self._update_embeddings,
            )
            self._free_keys_embedding = tf.get_variable(
                name="free_keys_embedding",
                dtype=tf.float32,
                shape=[self._n_keys, self._embedding_size],
                initializer=self._init,
                trainable=True,
            )

        self._nil_vars = set([self._embedding.name])

    def _mask_embedding(self, embedding):
        vocab_size, embedding_size = self._embedding_mat.shape
        embedding_mask = tf.constant(
            value=[0 if i == 0 else 1 for i in range(vocab_size)],
            shape=[vocab_size, 1],
            dtype=tf.float32,
            name="embedding_mask",
        )
        return embedding * embedding_mask

    def _inference(self, stories, endings, input_keep_prob, 
                       output_keep_prob, state_keep_prob, entnet_input_keep_prob, 
                       entnet_output_keep_prob, entnet_state_keep_prob, 
                       final_layer_keep_prob):
        with tf.variable_scope(self._name):
            masked_embedding = self._mask_embedding(self._embedding)

            batch_size = tf.shape(stories)[0]
            
            endings_emb = tf.nn.embedding_lookup(masked_embedding, endings)
            # [None, ending_size, emb_size]

            stories_emb = tf.nn.embedding_lookup(masked_embedding, stories)
            # [None, story_size, emb_size]

            stories_len = self._sentence_length(stories_emb)
            # [None]

            endings_len = self._sentence_length(endings_emb)
            # [None]

            stories_emb_shape = stories_emb.get_shape()
            sentence_rnn_cell_fw = tf.contrib.rnn.GRUCell(
                num_units=self._embedding_size
            )
            sentence_rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell=sentence_rnn_cell_fw,
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                variational_recurrent=True,
                input_size=(stories_emb_shape[2]),
                dtype=tf.float32,
            )
            sentence_rnn_cell_bw = tf.contrib.rnn.GRUCell(
                num_units=self._embedding_size
            )
            sentence_rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell=sentence_rnn_cell_bw,
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                variational_recurrent=True,
                input_size=(stories_emb_shape[2]),
                dtype=tf.float32,
            )
            (stories_output_fw, stories_output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=sentence_rnn_cell_fw,
                cell_bw=sentence_rnn_cell_bw,
                inputs=stories_emb,
                sequence_length=stories_len,
                dtype=tf.float32,
            )
            # stories_output_f/bw: [None, story_size, emb_size]
            # stories_state_f/bw: [None, emb_size]
            stories_output = stories_output_fw + stories_output_bw
            self._stories_output = stories_output
            # [None, story_size, emb_size]

            (_, _), (endings_state_fw, endings_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=sentence_rnn_cell_fw,
                cell_bw=sentence_rnn_cell_bw,
                inputs=endings_emb,
                sequence_length=endings_len,
                dtype=tf.float32,
            )
            # endings_output_f/bw: [None, ending_size, emb_size]
            # endings_state_f/bw: [None, emb_size]
            endings_state = endings_state_fw + endings_state_bw
            # [None, emb_size]
            
            free_keys_emb = self._free_keys_embedding
            # [n_keys, emb_size]

            keys_emb = tf.concat(
                values=[free_keys_emb],
                axis=0,
                name="keys_emb",
            )
            # [n_keys, emb_size]

            batched_keys_emb = tf.tile(
                input=tf.expand_dims(input=keys_emb, axis=0),
                multiples=[batch_size, 1, 1]
            )
            # [None, n_keys, emb_size]

            keys = tf.split(keys_emb, self._n_keys, axis=0)
            # list of [1, emb_size]
            keys = [tf.squeeze(key, axis=0) for key in keys]
            # list of [emb_size]

            alpha = tf.get_variable(
                name='alpha',
                shape=self._embedding_size,
                initializer=tf.constant_initializer(1.0)
            )
            activation = partial(prelu, alpha=alpha)

            cell_fw = DynamicMemoryCell(
                num_blocks=self._n_keys,
                num_units_per_block=self._embedding_size,
                keys=keys,
                initializer=self._init,
                recurrent_initializer=self._init,
                activation=activation,
            )
            initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            stories_output_shape = stories_output.get_shape()
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell=cell_fw,
                input_keep_prob=entnet_input_keep_prob,
                output_keep_prob=entnet_output_keep_prob,
                state_keep_prob=entnet_state_keep_prob,
                input_size=(stories_output_shape[2]),
                dtype=tf.float32,
            )

            cell_bw = DynamicMemoryCell(
                num_blocks=self._n_keys,
                num_units_per_block=self._embedding_size,
                keys=keys,
                initializer=self._init,
                recurrent_initializer=self._init,
                activation=activation,
            )
            initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            stories_output_shape = stories_output.get_shape()
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell=cell_bw,
                input_keep_prob=entnet_input_keep_prob,
                output_keep_prob=entnet_output_keep_prob,
                state_keep_prob=entnet_state_keep_prob,
                input_size=(stories_output_shape[2]),
                dtype=tf.float32,
            )
            (gate_output_fw, gate_output_bw), (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=stories_output,
                sequence_length=stories_len,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
            )
            # gate_output_f/bw: [None, story_size, n_keys]
            gate_output = (gate_output_fw + gate_output_bw) / 2
            # [None, story_size, n_keys]

            # last_state_f/bw: [None, (emb_size) * n_keys]
            last_state_fw = tf.stack(tf.split(last_state_fw, self._n_keys, axis=1), axis=1)
            # [None, n_keys, emb_size]
            last_state_bw = tf.stack(tf.split(last_state_bw, self._n_keys, axis=1), axis=1)
            # [None, n_keys, emb_size]


            last_state = last_state_fw + last_state_bw
            # [None, n_keys, emb_size]

            attention = tf.matmul(endings_state, tf.transpose(keys_emb, [1, 0]))
            # [None, n_keys]
            attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
            # [None, 1]
            attention = tf.nn.softmax(attention - attention_max)
            # [None, n_keys]
            attention = tf.expand_dims(attention, axis=2)
            # [None, n_keys, 1]

            u = tf.reduce_sum(last_state * attention, axis=1)
            # [None, emb_size] or [None, (emb_size) * 2]
            R = tf.get_variable('R', [self._embedding_size, 1])
            H = tf.get_variable('H', [self._embedding_size, self._embedding_size])

            q = endings_state
            # [None, emb_size]
            hidden = tf.nn.relu(q + tf.matmul(u, H))
            # [None, emb_size]
            hidden = tf.nn.dropout(x=hidden, keep_prob=final_layer_keep_prob)
            # [None, emb_size]
            y = tf.matmul(hidden, R)
            # [None, 1]

            return y, gate_output, stories_len

    def _get_mini_batch_start_end(self, n_train, batch_size=None):
        '''
        Args:
            n_train: int, number of training instances
            batch_size: int (or None if full batch)
        
        Returns:
            batches: list of tuples of (start, end) of each mini batch
        '''
        mini_batch_size = n_train if batch_size is None else batch_size
        batches = zip(
            range(0, n_train, mini_batch_size),
            list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
        )
        return batches

    def fit(self, stories, endings, answers, input_keep_prob, 
            output_keep_prob, state_keep_prob, entnet_input_keep_prob, 
            entnet_output_keep_prob, entnet_state_keep_prob, 
            final_layer_keep_prob, stories_feat, batch_size=None):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size)
            endings: Tensor (None, question_size)
            answers: Tensor (None)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        assert len(stories) == len(endings)
        assert len(stories) == len(answers)
        batches = self._get_mini_batch_start_end(len(stories), batch_size)
        total_loss = 0.
        for start, end in batches:
            feed_dict = {
                self._stories: stories[start:end], 
                self._endings: endings[start:end], 
                self._answers: answers[start:end], 
                self._input_keep_prob: input_keep_prob,
                self._output_keep_prob: output_keep_prob,
                self._state_keep_prob: state_keep_prob,
                self._entnet_input_keep_prob: entnet_input_keep_prob,
                self._entnet_output_keep_prob: entnet_output_keep_prob,
                self._entnet_state_keep_prob: entnet_state_keep_prob,
                self._final_layer_keep_prob: final_layer_keep_prob,
                self._stories_feat: stories_feat[start:end],
            }
            loss, _ = self._sess.run(
                [self.loss_op, self.train_op], 
                feed_dict=feed_dict
            )
            total_loss = loss * len(stories[start:end])
        return total_loss

    def predict(self, stories, endings, batch_size=None):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size)
            endings: Tensor (None, question_size)

        Returns:
            answers: Tensor (None)
        """
        assert len(stories) == len(endings)
        batches = self._get_mini_batch_start_end(len(stories), batch_size)
        predictions = []
        for start, end in batches:
            feed_dict = {
                self._stories: stories[start:end], 
                self._endings: endings[start:end], 
                self._input_keep_prob: 1.0,
                self._output_keep_prob: 1.0,
                self._state_keep_prob: 1.0,
                self._entnet_input_keep_prob: 1.0,
                self._entnet_output_keep_prob: 1.0,
                self._entnet_state_keep_prob: 1.0,
                self._final_layer_keep_prob: 1.0,
            }
            prediction = self._sess.run(
                self.predict_op,
                feed_dict=feed_dict
            )
            predictions.extend(prediction)
        return predictions
    
    def _sequence_loss(self, logits, labels, seq_len):
        '''
        Args:
            logits: [None, memory_size, 3]
            labels: [None, memory_size, 3]
            seq_len: [None]
        Returns:
            loss: []
        '''
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            name="cross_entropy_without_mask",
        )
        # [None, memory_size, 3]
        cross_entropy_mask = tf.sequence_mask(
            lengths=seq_len,
            maxlen=self._story_size,
            dtype=tf.float32,
        )
        # [None, memory_size]
        cross_entropy_mask = tf.expand_dims(cross_entropy_mask, -1)
        # [None, memory_size, 1]
        cross_entropy = tf.multiply(
            x=cross_entropy,
            y=cross_entropy_mask,
            name="cross_entropy_after_mask",
        )
        # [None, memory_size, 3]
        cross_entropy = tf.reduce_sum(
            input_tensor=cross_entropy,
            reduction_indices=1,
            name="cross_entropy_sum_sentence",
        )
        # [None, 3]
        cross_entropy = tf.div(
            x=cross_entropy,
            y=tf.expand_dims(tf.cast(x=seq_len, dtype=tf.float32), -1),
            name="cross_entropy_normalized_by_sent_len",
        )
        # [None, 3]
        cross_entropy = tf.reduce_sum(
            input_tensor=cross_entropy,
            reduction_indices=1,
            name="cross_entropy_sum_event_sentiment_topic"
        )
        loss = tf.reduce_mean(input_tensor=cross_entropy, name="loss")
        # loss: []
        return loss
    
    def _sentence_length(self, sentences):
        '''
        sentences: (None, sentence_len, embedding_size)
        '''
        used = tf.sign(tf.reduce_max(tf.abs(sentences), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
        
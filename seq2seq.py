import tensorflow as tf
import numpy as np
import time
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

START_ID = 0
PAD_ID = 1
END_ID = 2


# how does loss function work, should loss be a scalar?
# why shifted_target -2
# why use a moving avg in reward, is it similar to discount factor?
# is get unconflict even correct?
# gradient editing 梯度修剪 clip by norm
# embedding
# TODO Tensorboard
class MapperNet(object):
    """
    seq2seq network for cgra mapper
    """

    def __init__(self, batch_size=2, pea_size=4, max_input_seq_len=8, max_output_seq_len=10, input_vec_len=6,
                 rnn_size=32, attention_size=32, number_layers=1, beam_width=3,
                 learning_rate=0.001, max_gradient_norm=5, use_attention=False):

        self.batch_size = batch_size
        self.pea_size = pea_size
        self.rnn_size = rnn_size
        self.num_layers = number_layers
        self.max_input_seq_len = max_input_seq_len
        self.max_output_seq_len = max_output_seq_len
        self.input_vec_len = input_vec_len
        self.init_learning_rate = learning_rate
        self.use_attention = use_attention
        self.max_gradient_norm = max_gradient_norm
        self.beam_width = beam_width
        # Global step
        self.global_step = tf.Variable(0, trainable=False)
        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            # 添加dropout
            # cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=0.5)
            # dropout probability, see Recurrent Neural Network Regularization for reference
            return cell

        # 列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        output_layer = tf.layers.Dense(self.pea_size,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)) # 全连接层

        # Create placeholder
        self.placeholders = {
            'support': tf.sparse_placeholder(tf.float32),
            'features': tf.sparse_placeholder(tf.float32,
                                              shape=tf.constant([self.max_input_seq_len, self.max_input_seq_len],
                                                                dtype=tf.int64)),
        }
        self.advantage = tf.placeholder(tf.float32, shape=[self.batch_size], name="advantage")
        self.outputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_output_seq_len], name="outputs")
        self.enc_input_weights = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_input_seq_len],
                                                name="enc_input_weights")
        self.dec_input_weights = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_output_seq_len - 1],
                                                name="dec_input_weights")
        self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_input_seq_len, self.input_vec_len],
                                     name="inputs")
        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
        self.learning_rate = tf.placeholder('float32', None, name='learning_rate')
        # ======================== Two Layers of GCN =========================
        GraphConvolution_1 = GraphConvolution(input_dim=self.max_input_seq_len,
                                              output_dim=self.input_vec_len,
                                              placeholders=self.placeholders,
                                              act=lambda x: x,
                                              featureless=True,
                                              sparse_inputs=True)
        inputs = self.placeholders['features']
        outputGCN = GraphConvolution_1(inputs)
        # ======================== encoder define =========================
        # Calculate the lengths
        encoder_inputs = []
        for _ in range(self.batch_size):
            encoder_inputs.append(outputGCN)
        self.encoder_inputs = tf.stack(encoder_inputs)
        enc_input_lens = tf.reduce_sum(self.enc_input_weights, axis=1)
        dec_input_lens = tf.reduce_sum(self.dec_input_weights, axis=1) - 1
        self.max_batch_len = tf.reduce_max(enc_input_lens)

        # self.targets = tf.stack(self.output[1:], axis=1)
        enc_cell = self._create_rnn_cell()
        # encoder outputs and state
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(enc_cell, self.encoder_inputs, enc_input_lens, dtype=tf.float32)
        self.encoder_outputs = encoder_outputs
        self.encoder_state = encoder_state
        # Tile inputs if forward only
        # branch
        # Sized decoder cell
        # ======================== decoder define =========================
        dec_cell_0 = self._create_rnn_cell()
        if self.use_attention:
            # branch
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                                       memory_sequence_length=enc_input_lens)
            dec_cell = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell_0, attention_mechanism=attention_mechanism,
                                                           attention_layer_size=self.rnn_size, name='Attention_Wrapper')
            batch_size = self.batch_size
            decoder_initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)
            '''
            attention_mechanism_fw = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs_fw,
                                  memory_sequence_length=enc_input_lens_fw)
            dec_cell_fw = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell_0, attention_mechanism=attention_mechanism_fw, attention_layer_size=self.rnn_size, name='Attention_Wrapper')
            batch_size_fw = self.batch_size * self.beam_width
            decoder_initial_state_fw = dec_cell_fw.zero_state(batch_size=batch_size_fw, dtype=tf.float32).clone(cell_state=encoder_state_fw)
            '''
        else:
            decoder_initial_state = encoder_state
            # decoder_initial_state_fw = encoder_state_fw
        # branch
        self.decoder_initial_state = decoder_initial_state
        # self.decoder_initial_state_fw = decoder_initial_state_fw

        ##################################### forward inference #####################################
        shifted_START_ID = START_ID - 2
        shifted_END_ID = END_ID - 2
        embedding_lookup = np.array([[float(i)] for i in range(2, self.pea_size + 2)], dtype='float32')
        # embedding_lookup = np.array([[2.0],  # start id
        #                              [3.0],  # pea 0-0
        #                              [4.0],  # pea 0-0
        #                              [5.0],  # pea 0-1
        #                              [6.0],  # pea 0-0
        #                              [7.0],  # pea 0-0
        #                              [8.0],  # pea 0-0
        #                              [9.0],  # pea 0-1
        #                              [10.0],  # pea 0-0
        #                              [11.0],  # pea 0-0
        #                              [12.0],  # pea 0-0
        #                              [13.0],  # pea 0-1
        #                              [14.0],  # pea 0-0
        #                              [15.0],  # pea 0-0
        #                              [16.0],  # pea 0-0
        #                              [17.0],  # pea 0-1
        #                              [18.0]],
        #                             # [19.0],
                                    # [20.0],
                                    # [21.0],
                                    # [22.0],
                                    # [23.0],
                                    # [24.0],
                                    # [25.0],
                                    # [26.0],
                                    # [27.0],],  # pea 1-1
                                    # dtype='float32')
        self.start_tokens = tf.tile([START_ID], [self.batch_size]),
        # my_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding_lookup,
        #                                                start_tokens=tf.tile([START_ID],[self.batch_size]),
        #                                                end_token=0)
        my_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding=embedding_lookup,
                                                             start_tokens=tf.tile([START_ID], [self.batch_size]),
                                                             end_token=self.pea_size + 1,
                                                             softmax_temperature=1.0,
                                                             seed=int(time.time()))
        my_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                     my_helper,
                                                     decoder_initial_state,
                                                     output_layer=output_layer  # applied per timestep
                                                     )
        print('dynamic decode started....')
        # actor_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,  maximum_iterations=self.max_batch_len-2)
        actor_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                maximum_iterations=self.max_input_seq_len)  # maximum_iterations=self.max_batch_len+1)
        # actor_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder)
        self.actor_logits = actor_outputs.rnn_output
        self.infer_probs = tf.nn.softmax(self.actor_logits)
        predicted_ids = actor_outputs.sample_id
        # predicted_ids = actor_outputs.predicted_ids
        self.predicted_ids = predicted_ids
        #################################### backward update  #####################################
        self.outputs_list = tf.unstack(self.outputs, axis=1)
        decoder_inputs = tf.stack(self.outputs_list[:-2], axis=1)
        decoder_inputs = tf.reshape(decoder_inputs, [self.batch_size, self.max_input_seq_len, 1])
        self.decoder_inputs = decoder_inputs

        ## print node
        # decoder_inputs = tf.Print(decoder_inputs, [self.outputs_list, decoder_inputs], "***decoder input***", summarize=100)

        train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, dec_input_lens)
        # Basic Decoder
        train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, train_helper, decoder_initial_state, output_layer)
        # Decode
        train_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
        # logits
        # cur_batch_max_len = tf.reduce_max(dec_input_lens)
        cur_batch_max_len = self.max_input_seq_len
        # logits = output_layer(outputs.rnn_output)
        train_logits = train_outputs.rnn_output
        self.train_predicted_ids = train_outputs.sample_id
        # self.predicted_ids_with_logits=tf.nn.top_k(logits)
        # Pad logits to the same shape as targets
        # train_logits = tf.concat([train_logits,tf.ones([self.batch_size,
        #                                      self.max_output_seq_len-1-cur_batch_max_len,
        #                                      self.pea_size])],axis=1)
        # targets
        self.targets = tf.stack(self.outputs_list[1:-1], axis=1)
        self.targets = tf.cast(tf.reshape(self.targets, [self.batch_size, self.max_input_seq_len]), tf.int32)
        # self.shifted_targets = (self.targets-2) * self.dec_input_weights
        self.shifted_targets = self.targets - 2
        # this is negative log of chosen action
        self.train_logits = train_logits
        self.probs = tf.nn.softmax(train_logits)
        self.log_probs = tf.log(tf.nn.softmax(train_logits))
        self.neg_log_prob1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits,
                                                                            labels=self.shifted_targets)

        # self.neg_log_prob1 = tf.Print(self.neg_log_prob1, [self.shifted_targets, self.neg_log_prob1], \
        #         "train_logits and neg_log_prob1", summarize=100)

        # self.neg_log_prob2 = tf.reduce_sum(-tf.log(tf.nn.softmax(train_logits, name='act_prob')) * tf.one_hot(self.shifted_targets, self.pea_size), axis=2)
        self.neg_log_prob2 = tf.reduce_sum(
            tf.nn.softmax(train_logits, name='act_prob') * tf.one_hot(self.shifted_targets, self.pea_size), axis=2)
        # self.neg_log_prob2 = -tf.log(tf.nn.softmax(train_logits, name='act_prob')) * tf.one_hot(self.shifted_targets, self.pea_size+1)

        # self.log_probs = tf.transpose(log_probs,[1,0])
        # self.neg_log_probs1 = tf.reduce_sum((self.neg_log_prob1 * tf.cast(self.dec_input_weights, dtype=tf.float32)), axis=1)
        self.neg_log_probs2 = tf.reduce_sum(self.neg_log_prob1, axis=1)
        # self.neg_log_probs2 = self.neg_log_probs1 / tf.cast(tf.reduce_sum(self.dec_input_weights, axis=1), dtype=tf.float32)
        # self.neg_log_probs2 = self.neg_log_probs1

        mean_neg_log_probs = tf.reduce_mean(self.neg_log_probs2)
        mean_advantage = tf.reduce_mean(self.advantage)
        # self.neg_log_probs2 = tf.reduce_sum(self.neg_log_prob1, axis=1)

        reinforce = self.advantage * self.neg_log_probs2
        self.actor_loss = tf.reduce_mean(reinforce)

        # self.actor_loss = tf.Print(self.actor_loss, [self.actor_loss, reinforce], \
        #         "no====", summarize=100)
        self.learning_rate_op = self.learning_rate
        # self.learning_rate_op = tf.maximum(1e-6,
        #     tf.train.exponential_decay(
        #        self.init_learning_rate,
        #        self.learning_rate_step,
        #        1000,
        #        0.9,
        #        staircase=True))
        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01)

        optimizer = tf.train.AdamOptimizer(self.init_learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.init_learning_rate, beta1=0.8, beta2=0.888, epsilon=1e-08)
        grads_and_vars = optimizer.compute_gradients(self.actor_loss)
        for idx, (grad, var) in enumerate(grads_and_vars):
            if grad is not None:
                grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_gradient_norm), var)
        self.actor_update = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # actor_optimizer = tf.train.AdamOptimizer(0.001)
        # self.actor_update = actor_optimizer.minimize(self.actor_loss)
        # Get all trainable variables
        parameters = tf.trainable_variables()
        # Calculate gradients
        # gradients = tf.gradients(self.actor_loss, parameters)
        # Clip gradients
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # Optimization
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.init_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01)
        # Update operator
        # self.actor_update = optimizer.apply_gradients(zip(clipped_gradients, parameters),global_step=self.global_step)
        # Summarize
        tf.summary.scalar('loss', self.actor_loss)
        tf.summary.scalar('nlog_probs', mean_neg_log_probs)
        tf.summary.scalar('advantage', mean_advantage)
        # tf.summary.scalar('learning_rate', self.learning_rate_op)
        for p in parameters:
            tf.summary.histogram(p.op.name, p)
        # for p in gradients:
        #  tf.summary.histogram(p.op.name,p)
        # Summarize operator
        self.summary_op = tf.summary.merge_all()
        # Save
        self.saver = tf.train.Saver(tf.global_variables())
        ##################################### backward inference #####################################
        '''
        outputs_list = tf.unstack(self.outputs, axis=1)
        decoder_inputs = tf.stack(outputs_list[:-1],axis=1)
        decoder_inputs = tf.reshape(decoder_inputs, [self.batch_size, self.max_output_seq_len-1, 1])
        self.decoder_inputs = decoder_inputs
        # Training Helper
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, dec_input_lens)
        # Basic Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, decoder_initial_state, output_layer)
        # Decode
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        # logits
        cur_batch_max_len = tf.reduce_max(dec_input_lens)
        #logits = output_layer(outputs.rnn_output)
        logits = outputs.rnn_output
        self.predicted_ids_with_logits=tf.nn.top_k(logits)
        # Pad logits to the same shape as targets
        logits = tf.concat([logits,tf.ones([self.batch_size,
                                              self.max_output_seq_len-1-cur_batch_max_len,
                                              self.pea_size+1])],axis=1)
        self.logits = logits
        # targets
        self.targets = tf.stack(outputs_list[1:],axis=1)
        self.targets = tf.cast(tf.reshape(self.targets, [self.batch_size, self.max_output_seq_len-1]), tf.int32)
        self.shifted_targets = (self.targets-2) * self.dec_input_weights
        # Losses
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_targets, logits=logits)
        # Total loss
        self.loss = tf.reduce_sum(self.losses*tf.cast(self.dec_input_weights,tf.float32))/self.batch_size
        # Get all trainable variables
        parameters = tf.trainable_variables()
        # Calculate gradients
        gradients = tf.gradients(self.loss, parameters)
        # Clip gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # Optimization
        optimizer = tf.train.AdamOptimizer(self.init_learning_rate)
        # Update operator
        self.update = optimizer.apply_gradients(zip(clipped_gradients, parameters),global_step=self.global_step)
        # Summarize
        tf.summary.scalar('loss',self.loss)
        for p in parameters:
          tf.summary.histogram(p.op.name,p)
        for p in gradients:
          tf.summary.histogram(p.op.name,p)
        # Summarize operator
        self.summary_op = tf.summary.merge_all()
        #DEBUG PART
        self.debug_var = logits
        '''

    def optim(self, session, adj, features, inputs, input_weights, advantage, actions, action_weights, steps, learning_rate):
        """
        Run a step
        """
        input_feed = {}
        input_feed[self.placeholders["support"]] = adj
        input_feed[self.placeholders["features"]] = features
        input_feed[self.inputs] = inputs
        input_feed[self.advantage] = advantage
        input_feed[self.outputs] = actions
        input_feed[self.enc_input_weights] = input_weights
        input_feed[self.dec_input_weights] = action_weights
        input_feed[self.learning_rate_step] = steps
        # if steps % 10000 < 5000:
        #  lr = 1e-4 + 9e-4 * (steps % 5000) / 5000
        # else:
        #  lr = 1e-3 - 9e-4 * ((steps-5000) % 5000) / 5000
        input_feed[self.learning_rate] = learning_rate

        output_feed = [self.actor_update, self.actor_loss, self.outputs, self.outputs_list, self.decoder_inputs, \
                       self.targets, self.shifted_targets, self.train_logits, self.probs, self.train_predicted_ids, \
                       self.log_probs, self.neg_log_prob1, self.neg_log_prob2, self.neg_log_probs2, self.summary_op,
                       self.learning_rate_op]
        results = session.run(output_feed, input_feed)
        return results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[
            9], results[10], results[11], results[12], results[13], results[14], results[15]

    def step(self, session, adj, features, inputs, input_weights, outputs=None, output_weights=None, forward_only=False):
        """
        Run a step
        """
        input_feed = {}
        input_feed[self.placeholders["support"]] = adj
        input_feed[self.placeholders["features"]] = features
        input_feed[self.inputs] = inputs
        input_feed[self.enc_input_weights] = input_weights
        if forward_only == False:
            input_feed[self.outputs] = outputs
            input_feed[self.dec_input_weights] = output_weights

        if forward_only:
            # output_feed = [self.predicted_ids, self.start_tokens, self.decoder_initial_state]
            output_feed = [self.max_batch_len, self.predicted_ids, self.infer_probs, self.actor_logits, self.encoder_inputs]
        else:
            output_feed = [self.update, self.summary_op, self.loss, self.predicted_ids_with_logits,
                           self.shifted_targets, self.debug_var,
                           self.encoder_outputs, self.encoder_state, self.decoder_inputs, self.dec_input_weights,
                           self.encoder_state]

        # Run step
        results = session.run(output_feed, input_feed)

        if forward_only:
            # print('max_batch_len', results[0])
            # print('predicted_ids', results[1])
            return results[1], results[0], results[2], results[3], results[4]
        else:
            print('inputs is: ', inputs)
            print('outputs is: ', outputs)
            print('decoder inputs is: ', results[8])
            print('dec inputs weights is: ', results[9])
            print('shifted targets is: ', results[4])
            return results[1], results[2], results[3], results[4], results[5]

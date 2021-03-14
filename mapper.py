import tensorflow as tf
import seq2seq as seq2seq_model
from cgra_info import pea_4x4_routes
import time
import os
from utils import *
from graph import *

FLAGS = tf.app.flags.FLAGS
np.set_printoptions(threshold=np.inf)
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size.")
tf.app.flags.DEFINE_integer("input_vec_len", 4, "Input vector length.")
tf.app.flags.DEFINE_integer('hidden1', 6, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_string("data_path", "data/si2_b03_m400.A10.out", "The path to the file including source graph.")
# tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
log_dir = "./log"
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "frequence to do per checkpoint.")
tf.app.flags.DEFINE_boolean("forward_only", True, "Forward Only.")
tf.app.flags.DEFINE_boolean("use_attention", True, "Use attention.")
tf.app.flags.DEFINE_integer("ii", 2,"Initiation interval.")
tf.app.flags.DEFINE_string("target_graph_path", "data/si2_b03_m400.B10.out", "The path to the file inluding target graph")
tf.app.flags.DEFINE_boolean("verbose", False, "vebose output")
tf.app.flags.DEFINE_boolean("continue_training", True, "whether continue training after a match is found")
tf.app.flags.DEFINE_integer("max_iteration", 10, "force the solver to stop after this number of iterations, negitive value will disable it")
tf.app.flags.DEFINE_integer("max_input_seq_len", 80, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_seq_len", 82, "Maximum output sequence length.")

class Mapper(object):
    def __init__(self, forward_only):
        self.forward_only = forward_only
        self.graph = tf.Graph()
        self.succeed_times = 0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        with self.graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            ## self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        #储存为邻接表形式
        self.target_graph = TargetGraph(FLAGS.target_graph_path)
        self.source_graph = SourceGraph(FLAGS.data_path)

        tf.app.flags.DEFINE_integer("pea_size", self.target_graph.get_grf_size(), "The number of PEs in the PEA.")
        self.read_data()
        self.build_model()
        self.pea_size = FLAGS.pea_size    
        self.ii = FLAGS.ii
        self.node2pe_map = []
        self.node2ts_map = []
        self.node2id_map = []

    def read_data(self):
        with open(FLAGS.data_path, 'r') as file:
            recs = file.readlines()
            inputs = []
            enc_input_weights = []
            outputs = []
            dec_input_weights = []

        adj = ReadingGraph(FLAGS.data_path).dt   #邻接矩阵
        adj = preprocess_adj(adj)
        feature = sp.identity(adj[2][0])
        feature = preprocess_features(feature)
        for rec in recs:
            # print('-'*60)
            if (rec[-1] == '\n'):
                inp, outp = rec[:-1].split(' output ')
            else:
                inp, outp = rec.split(' output ')

            inp = inp.split(',')
            outp = outp.split(',')

            enc_input = []
            for t in inp:
                enc_input.append(float(t))
            enc_input_len = len(enc_input) // FLAGS.input_vec_len
            enc_input += [0] * ((FLAGS.max_input_seq_len - enc_input_len) * FLAGS.input_vec_len)
            enc_input = np.array(enc_input).reshape([-1, FLAGS.input_vec_len])
            inputs.append(enc_input)
            weight = np.zeros(FLAGS.max_input_seq_len)
            weight[:enc_input_len] = 1
            enc_input_weights.append(weight)
            if FLAGS.verbose == True:
                print('input weight:', weight)

            output = [seq2seq_model.START_ID]
            for i in outp:
                output.append(int(i) + 2)
            output.append(seq2seq_model.END_ID)
            dec_input_len = len(output)
            output += [seq2seq_model.PAD_ID] * (FLAGS.max_output_seq_len - dec_input_len)
            output = np.array(output)
            outputs.append(output)
            weight = np.zeros(FLAGS.max_output_seq_len - 1)
            weight[:dec_input_len - 1] = 1
            dec_input_weights.append(weight)
            if FLAGS.verbose == True:
                print('output weight:', weight)
                print('input:',enc_input)
                print('output:',output)

        self.inputs = np.stack(inputs)
        self.enc_input_weights = np.stack(enc_input_weights)
        self.outputs = np.stack(outputs)
        self.dec_input_weights = np.stack(dec_input_weights)
        self.adj = adj
        self.features = feature
        if FLAGS.verbose == True:
            print("self dec input weights:", self.dec_input_weights)
            print("Load inputs:            " + str(self.inputs.shape))
            print("Load enc_input_weights: " + str(self.enc_input_weights.shape))
            print("Load outputs:           " + str(self.outputs.shape))
            print("Load dec_input_weights: " + str(self.dec_input_weights.shape))
            print("Load adjacency_matrix: ", self.adj)
            print("Load features: ", self.features)

    def build_model(self):
        with self.graph.as_default():
            # Build Model
            self.model = seq2seq_model.MapperNet(
                batch_size=FLAGS.batch_size,
                pea_size=FLAGS.pea_size,
                max_input_seq_len=FLAGS.max_input_seq_len,
                max_output_seq_len=FLAGS.max_output_seq_len,
                input_vec_len=FLAGS.input_vec_len,
                rnn_size=96,
                number_layers=6,
                learning_rate=1e-5,
                max_gradient_norm=5,
                use_attention=FLAGS.use_attention)
            # Prepare Summary writer
            self.writer = tf.summary.FileWriter(log_dir + '/train', self.sess.graph)
            # Try to get checkpoint(continious learning)
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
              print("Load model parameters from %s" % ckpt.model_checkpoint_path)
              self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
              print("Created model with fresh parameters.")
              self.sess.run(tf.global_variables_initializer())

    # 1. fix batch
    # 2. fix floyd
    # 4. try penalize
    # 3. why ** 1/3
    # implement vf2 rules into helper function(get unconflict)
    # try to ease the helper function when vf2 rule is too strict
    def reward_revised(self, inputs, actions, current_step, average_step):
        distance_sum = np.zeros(FLAGS.batch_size, dtype=float)
        rewards = np.zeros(FLAGS.batch_size, dtype=float)

        for batch in range(FLAGS.batch_size):
            assert len(actions[batch]) == len(inputs[batch])
            node_size = len(inputs[batch])
            for node in inputs[batch]:
                node_idx = int(node[0])
                target_node = actions[batch][node_idx - 1]
                for neighbour_idx in node[1:]:
                    neighbour_idx = int(neighbour_idx)
                    if neighbour_idx == 0:
                        continue
                    neighbour_target_node = int(actions[batch][neighbour_idx - 1])
                    if self.target_graph.get_distance_between(target_node + 1, neighbour_target_node + 1) != 1: # not connected
                        distance_sum[batch] += self.target_graph.get_distance_between(target_node + 1, neighbour_target_node + 1) # aggregated distance

            rewards[batch] = - (distance_sum[batch] ** (1 / 2))

        action_num = [len(batch) for batch in actions]
        with open("best_reward.txt", "a") as f:
            idx = np.argmax(rewards)
            line = ",".join([str(n) for n in actions[idx].tolist()])
            f.write(line + "\n")
        return rewards, action_num

 

    def get_unconflict_prediction_revised(self, inputs, actor_logits):
        #check whether start from 0
        predicted_ids = np.ones((FLAGS.batch_size, FLAGS.max_input_seq_len), dtype=np.int) * FLAGS.pea_size

        for batch in range(FLAGS.batch_size):
            new_predicted_ids = []    #已匹配的结点
            new_predicted_probs = []
            for i in range(FLAGS.max_input_seq_len):
                # give me an list of preceeding actions, return the feasible mapped nodes
                # set the weight on all infeasible actions to -inf
                infeasible = self.target_graph.get_infeasible_mapping_node(self.source_graph, new_predicted_ids, i == 0)
                
                if len(infeasible) == len(self.target_graph.graph):        #全部不可匹配？
                    # give up
                    infeasible = new_predicted_ids
                for node in infeasible:
                    actor_logits[batch][i][node] = -np.inf    #匹配目标图中节点概率
                #infeasible = new_predicted_ids

                # sofmax and sample a new action
                new_logits = np.array(actor_logits[batch][i])
                new_logits = new_logits - np.max(new_logits)

                probs = np.exp(new_logits) / np.sum(np.exp(new_logits))

                action = np.random.choice(FLAGS.pea_size, 1, p=probs)

                new_predicted_ids.append(action[0])
                new_predicted_probs.append(probs[action[0]])
            predicted_ids[batch] = new_predicted_ids
        return predicted_ids


    def eval(self):
        """ Randomly get a batch of data and output predictions """
        step_time = 0.0
        loss = 0.0
        epoch_rewards = 0.0
        last_epoch_rewards = -100.0
        current_step = 0
        average_step = 0
        global start_time, end_time
        critic_exp_mvg_avg = np.zeros(1)
        learning_rate = 1e-5
        start_time = time.time()
        while True:
            print('*' * 100)
            print('**', ' ' * 43, 'START', ' ' * 44, '**')
            print('*' * 100)
            print('Step is: ', current_step)
            print('\n1. Getting batch data...')
            print('------------------------')
            # get_batch的作用是将一组数据复制16遍形成一个batch
            inputs, enc_input_weights, outputs, dec_input_weights = self.get_batch()

            if FLAGS.verbose == True:
                print('inputsbefore', inputs)
            # for batch in range(FLAGS.batch_size):
            #     for node in range(inputs[batch].size // FLAGS.input_vec_len):
            #         node_ts = int(inputs[batch][node][1])
            #         # inputs[batch][node][1] = node_ts % FLAGS.ii
            #         inputs[batch][node][1] = node_ts
            # self.shuffle(inputs)
            outpus = tf.zeros(shape=[FLAGS.batch_size, FLAGS.max_output_seq_len], dtype=tf.float32)
            if FLAGS.verbose == True:
                print('input weights:\n', enc_input_weights)
            # get actions
            print('\n2. Choose action on policy...')
            print('-----------------------------')
            o_predicted_ids, max_nodes_num, infer_probs, actor_logits, outputsGCN = self.model.step(session=self.sess,
                                                                                                    adj=self.adj,
                                                                                                    features=self.features,
                                                                                                    inputs=inputs,
                                                                                                    input_weights=enc_input_weights,
                                                                                                    outputs=outputs,
                                                                                                    output_weights=dec_input_weights,
                                                                                                    forward_only=True)

            if FLAGS.verbose == True:
                print('max_nodes_num:\n', max_nodes_num)
                print('infer probs:\n', infer_probs)
                print('actor logits:\n', actor_logits)

            print('predicted action:', o_predicted_ids)
            predicted_ids = self.get_unconflict_prediction_revised(inputs, actor_logits)
            print("unconflict pr", predicted_ids)
            # predicted_ids = np.ones((FLAGS.batch_size, FLAGS.max_input_seq_len), dtype=np.int)*FLAGS.pea_size
            # predicted_ids[0] = np.array([4, 13,  4, 11,  3,  5, 11, 11,  5,  4,  6,  3,  3, 10, 10,  5,  6, 10,  7], dtype=np.int)
            print('\n3. Calculate rewards...')
            print('-----------------------')
            rewards, action_num = self.reward_revised(inputs, predicted_ids, current_step, average_step)

            with open("reward_file.log", "a") as f:
                line = str(current_step) + "," + str(np.mean(rewards)) + "," + str(np.min(rewards)) + "," + str(np.max(rewards)) + "\n"
                f.write(line)

            if FLAGS.continue_training == False:
                for batch in range(FLAGS.batch_size):
                    if abs(rewards[batch] - 0) < 1e-5:
                        end_time = time.time()
                        print(rewards)
                        with open("out.txt", "a") as f:
                            f.write("success, cost " + str(-(start_time-end_time)) + " seconds\n")
                            f.write("the action is:" + str(predicted_ids[batch] + 1))
                            f.write("the current step is:" + str(current_step) + "\n")
                        os._exit(0)

            if FLAGS.max_iteration > 0 and current_step >= FLAGS.max_iteration:
                print("reward:", rewards)
                print("actions:", predicted_ids)
                os._exit(0)

            print('\n4. Calculate formed action and dec input weights...')
            print('---------------------------------------------------')
            action_list = np.zeros((FLAGS.batch_size, FLAGS.max_output_seq_len), dtype=np.int)
            for batch in range(FLAGS.batch_size):
                if predicted_ids[batch].size > FLAGS.max_output_seq_len - 1:
                    aligned_predicted_ids = predicted_ids[batch][0:FLAGS.max_output_seq_len - 1]
                else:
                    aligned_predicted_ids = predicted_ids[batch]

                action_list[batch] = np.concatenate([[0], \
                                                     aligned_predicted_ids + 2, \
                                                     np.ones(FLAGS.max_output_seq_len - 1 - aligned_predicted_ids.size,
                                                             dtype=np.int32)])
                # dec_input_weights[batch] = np.concatenate([np.ones(predicted_ids[batch].size, dtype=np.int32), \
                #                                           np.zeros(FLAGS.max_output_seq_len-predicted_ids[batch].size-1, dtype=np.int32)])
                valid_action_num = min(action_num[batch], FLAGS.max_output_seq_len - 2)
                dec_input_weights[batch] = np.concatenate([np.ones(valid_action_num + 1, dtype=np.int32), \
                                                           np.zeros(FLAGS.max_output_seq_len - valid_action_num - 2,
                                                                    dtype=np.int32)])
            if FLAGS.verbose == True:
                print("action_list:\n", action_list)
                print('output weights:\n', dec_input_weights)

            print('\n5. Calculate critic_exp_mvg')
            print('---------------------------')
            if current_step == 0:
                critic_exp_mvg_avg = np.mean(rewards)
            else:
                # critic_exp_mvg_avg = critic_exp_mvg_avg*0.9 + np.mean(rewards)*0.1
                critic_exp_mvg_avg = critic_exp_mvg_avg * 0.99 + np.mean(rewards) * 0.01
            print('\n6. Calculate advantage')
            print('--------------------------')
            # advantage = rewards - critic_exp_mvg_avg
            if not np.max(rewards) == np.min(rewards):
                advantage = (rewards - np.mean(rewards)) / (np.max(rewards) - np.min(rewards))
            else:
                advantage = rewards - critic_exp_mvg_avg
            # advantage = (rewards - np.mean(rewards)) / (np.max(rewards) - np.min(rewards))
            # advantage = rewards
            print('\n7. Update the netrwork')
            print('--------------------------')
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            step_loss, outputs1, outputs_list, decoder_inputs, targets, shifted_targets, logits, probs, train_ids, log_probs, \
            neg_log_prob1, neg_log_prob2, neg_log_probs2, summary, lr \
                = self.model.optim(session=self.sess,
                                   inputs=inputs,
                                   input_weights=enc_input_weights,
                                   advantage=advantage,
                                   actions=action_list,
                                   action_weights=dec_input_weights,
                                   steps=current_step,
                                   learning_rate=learning_rate,
                                   adj=self.adj,
                                   features=self.features)

            with open("loss.log", "a") as f:
                line = str(current_step) + "," + str(step_loss) + "\n"
                f.write(line)

            loss += step_loss / FLAGS.steps_per_checkpoint
            if FLAGS.verbose == True:
                print('outputs:\n', outputs1)
                # print('outputs_list:\n', outputs_list)
                # print('decoder_inmputs:\n', decoder_inputs)
                print('targets:\n', targets)
                # print('logits:\n', logits)
                # print('probs:\n', probs)
                print('train_ids:\n', train_ids)
                # print('logprobs:\n', log_probs)
                print('shifted_targets:\n', shifted_targets)
                # print('neg_log_prob1:\n', neg_log_prob1)
                # print('neg_log_prob2:\n', neg_log_prob2)
                # print('neg_log_probs1:\n', neg_log_probs1)
                # print('lg rate: ', lr)
                # epoch_rewards += np.mean(rewards) / FLAGS.steps_per_checkpoint
                print('neg_log_probs2:\n', neg_log_probs2)
                print('advantage:', advantage)

            print('(mean rewards):', current_step, ",", np.mean(rewards), ",", np.max(rewards))
            print('rewards is', rewards)
            print('step_loss: ', step_loss)

            epoch_rewards += np.mean(rewards)
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                epoch_rewards = epoch_rewards / FLAGS.steps_per_checkpoint
                print("epoch_rewards, last_epoch_rewards:", epoch_rewards, ",", last_epoch_rewards)
                learning_rate = learning_rate * 0.95
                if epoch_rewards < last_epoch_rewards:
                    learning_rate = learning_rate * 0.5
                last_epoch_rewards = epoch_rewards
                with self.sess.as_default():
                    gstep = self.model.global_step.eval()
                print("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))
                # Write summary
                self.writer.add_summary(summary, gstep)
                checkpoint_path = os.path.join(log_dir, "convex_hull.ckpt")
                self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
                step_time, loss = 0.0, 0.0
                epoch_rewards = 0.0

    def get_batch(self):
        data_size = self.inputs.shape[0]
        print('data size is: ', data_size)
        sample = np.random.choice(data_size, FLAGS.batch_size, replace=True)
        print('sample is: ', sample)
        return self.inputs[sample], self.enc_input_weights[sample],\
            self.outputs[sample], self.dec_input_weights[sample]


def main(_):
    print('Main stared ...')
    mapper = Mapper(FLAGS.forward_only)
    if FLAGS.forward_only:
      mapper.eval()
    # else:
    #   mapper.train()

if __name__ == "__main__":
    tf.app.run()


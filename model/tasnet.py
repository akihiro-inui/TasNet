import tensorflow as tf
from model.parameters import *
from tensorflow.keras.layers import LayerNormalization, Dense
from itertools import product, permutations


class TasNet(object):

    def __init__(self,  batch_size, frame_length):
        """
        :param batch_size: batch_size
        :param frame_length: One frame length
        """
        self.rnn_hidden = rnn_hidden_size
        self.K = int(frame_length)
        self.context = 3
        # Floor the number to the closest integer (half of window size)
        self.context_window = self.context // 2
        self.nspk = 2  # Number of speakers
        self.batch_size = batch_size

        # Random variables
        self.U = tf.Variable(tf.random.truncated_normal(shape=[L, 1, N], dtype=tf.float32, name='U'))
        self.V = tf.Variable(tf.random.truncated_normal(shape=[L, 1, N], dtype=tf.float32, name='V'))
        self.B = tf.Variable(tf.random.truncated_normal(shape=[N, L], dtype=tf.float32, name='B'))

    def BLSTM_layernorm(self, input, index):
        var_scope = 'BLSTM' + str(index)
        with tf.variable_scope(var_scope) as scope:
            lstm_fw_cell = tf.keras.rnn.LayerNormBasicLSTMCell(
                self.rnn_hidden, layer_norm=True, )
            lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.rnn_hidden, layer_norm=False, )
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell, input,
                sequence_length=[self.context * N] * self.batch_size,
                dtype=tf.float32)
            output = tf.concat(outputs, 2)
        return output

    def BLSTM(self, input, index):
        var_scope = 'BLSTM' + str(index)
        with tf.variable_scope(var_scope) as scope:
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(
                self.rnn_hidden, use_peepholes=True, cell_clip=25, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(
                self.rnn_hidden, use_peepholes=True, cell_clip=25, state_is_tuple=True)
            initial_fw = lstm_fw_cell.zero_state(tf.shape(input)[0], dtype=tf.float32)
            initial_bw = lstm_bw_cell.zero_state(tf.shape(input)[0], dtype=tf.float32)
            output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                                        sequence_length=[self.K]*self.batch_size,
                                                        initial_state_fw=initial_fw,
                                                        initial_state_bw=initial_bw,
                                                        dtype=tf.float32,
                                                        time_major=False)
            output = tf.concat(output, 2)
        return output

    def encoder(self, speech_mixture):
        """
        Encoder
        It takes mixture audio vectors as input and returns mixture weight
        Calculate mixture weight matrix by following steps;
            1. L2 normalization to input mixture signal
            2. 1D gated convolution along time dimension
            3. Apply non-linearity (3.1. sigmoid, 3.2. ReLU)
            4. Multiply 3.1 and 3.2

        Args:
            :param speech_mixture: Input mixture signals [B, K, L]
            :return mixture_weight: Non-negative mixture weight matrix [B, K, N]
            :return norm_coefficient: Normalize coefficient [B, K, 1]

        B: Number of items (number of audio files)
        K: Number of frames from one audio
        L: One frame length
        N: Number of basis
        """
        # Get input batch data size
        B, K, L = speech_mixture.size()

        # 1, L2 Norm along L axis
        norm_coefficient = tf.sqrt(tf.reduce_sum(speech_mixture ** 2, axis=2, keepdims=True))
        norm_mixture = speech_mixture / (norm_coefficient + eps)  # B x K x L

        # 2. Flatten all items to 2D matrix
        norm_mixture = tf.expand_dims(tf.reshape(norm_mixture, [-1, L]), axis=2)

        # 3.1 Apply relu [B*K, L, 1] conv [L, 1, N] -> [B*K, N]
        conv = tf.nn.relu(tf.nn.conv1d(norm_mixture, self.U, stride=1, padding='VALID'))

        # 3.2 Apply sigmoid [B*K, L, 1] conv [L, 1, N] -> [B*K, N]
        gate = tf.nn.sigmoid(tf.nn.conv1d(norm_mixture, self.V, stride=1, padding='VALID'))

        # 4 Multiply output matrices from sigmoid and ReLU [B*K,N]
        mixture_weight = conv * gate  # B*K x N x 1
        mixture_weight = tf.reshape(mixture_weight, [self.batch_size, -1, N])  # [B, K, N]

        self.summary_conv = tf.summary.histogram('encoder_conv', conv)
        self.summary_gate = tf.summary.histogram('encoder_gate', gate)

        return mixture_weight, norm_coefficient

    def separator(self, mixture_weight):
        """
        Separation Network
        It takes mixture weight as input and returns mask
        Calculate mask by following steps;
            1. Layer normalization to input mixture weight
            2. Zero padding
            3. BLSTM layer
            4. Fully Connected layer
        Args:
            :param mixture_weight: [B, K, N]
            :return mask_fc: [B, K, nspk, N]
        """
        # 1. Layer normalization [B, K, N]
        norm_mixture_weight = LayerNormalization(mixture_weight)
        norm_mixture_weight = tf.reshape(norm_mixture_weight, (self.batch_size, self.K, N))

        self.summary_layer_norm_mix = tf.summary.histogram('separator_layer_norm_mix_w', norm_mixture_weight)

        # 2. Zero padding -> [B, K, context * N]
        blank_ = tf.zeros([self.batch_size, self.context_window, N], dtype=tf.float32)
        # [B, context_window + K + context_window, N]
        padded_w_ = tf.concat([blank_, norm_mixture_weight, blank_], axis=1)
        frame_index = 0
        new_w_ = padded_w_[:, frame_index: frame_index + self.context, :]
        for frame_index in range(1, self.K):
            new_w_ = tf.concat([new_w_,
                                padded_w_[:, frame_index: frame_index + self.context, :]],
                               axis=1)
        contexted_w = tf.reshape(new_w_, [self.batch_size, self.K * self.context, N])
        contexted_w = tf.reshape(contexted_w, [self.batch_size, self.K, self.context * N])

        # 3. BLSTM layer [B*K, rnn_layer_size]
        lstm1 = self.BLSTM(contexted_w, 1)
        lstm2 = self.BLSTM(lstm1, 2)
        lstm3 = self.BLSTM(lstm2, 3)
        lstm4 = self.BLSTM(lstm3 + lstm2, 4)
        output = lstm4  # [B, hidden]
        lstm_out = tf.reshape(output, [-1, 2 * self.rnn_hidden])  # [B*K, 2 * rnn_hidden]
        self.summary_lstm_out = tf.summary.histogram('separator_lstm_out', lstm_out)

        # 4. Fully Connected layer [B, K, nspk, N]
        fc = Dense(lstm_out, self.nspk*N, activation=None)
        mask_fc = tf.reshape(fc, [self.batch_size, self.K, self.nspk, N])
        mask_fc = tf.nn.softmax(mask_fc, axis=2)
        self.summary_lstm_out = tf.summary.histogram('separator_lstm_out', lstm_out)

        return mask_fc

    def decoder(self, mixture_weight, est_mask):
        """
        Decoder
        :param mixture_weight: [B, K, N]
        :param est_mask: [B, K, nspk, N]
        :return: est_source: [B, K, nspk, L]
        """
        source_w = est_mask * tf.expand_dims(mixture_weight, axis=2)  # [B, K, nspk, N]

        # source_w [B, K, nspk, N], var_B [N, L] -> [B, K, nspk, L]
        # Get estimated source as time domain signal
        estimated_source = tf.einsum('bkcn,nl->bkcl', source_w, self.B)

        self.summary_B = tf.summary.histogram('decoder_basis_signals', self.B)
        return estimated_source

    def build_network(self, mixture):
        # Input: Speech mixture [B, K, L], Output: Mixture weight [B, K, N], normalize coefficient [B, K, 1]
        mixture_weight, norm_coefficient = self.encoder(mixture)
        # Input: Mixture weight [B, K, N] , Output: Estimated mask [B, K, nspk, N]
        estimated_mask = self.separator(mixture_weight)
        # Input: Mixture weight [B, K, N] and estimated mask [B, K, nspk, N], Output: Estimated source [B, K, nspk, L]
        estimated_source = self.decoder(mixture_weight, estimated_mask)  # [B, K, nspk, L]

        norm_coef_ = tf.expand_dims(norm_coefficient, axis=2)  # [B, K, 1, 1]
        est_source = tf.transpose(estimated_source * norm_coef_, [0, 2, 1, 3])    # [B, nspk, K, L]

        return est_source

    def train(self, loss, lr):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # optimizer = tf.segment_test.MomentumOptimizer(lr, 0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 200)
            train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op

    def objective(self, est_source, source):
        """
        :param est_source: [B, C, K, L]
        :param source: [B, C, K, L]
        :return:
        """
        max_snr, v_perms, max_snr_idx = self.get_si_snr(source, est_source)
        loss = 20 - tf.reduce_mean(max_snr)
        tar_perm = tf.gather(v_perms, max_snr_idx)
        tar_perm = tf.transpose(tf.one_hot(tar_perm, self.nspk), [0, 2, 1])
        tar_perm = tf.cast(tf.argmax(tar_perm, axis=2), tf.int32)
        outer_axis = tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1]), [1, self.nspk])
        gather_idx = tf.stack([outer_axis, tar_perm], axis=2)
        gather_idx = tf.reshape(gather_idx, [-1, 2])
        reorder_recon = tf.reshape(tf.gather_nd(est_source, gather_idx),
                                   [self.batch_size, self.nspk, -1, L])

        self.loss_summary = tf.summary.scalar('tasnet_loss', loss)
        self.snr_summary = tf.summary.scalar('snr', tf.reduce_mean(max_snr))
        return loss, max_snr, est_source, reorder_recon

    def get_si_snr(self, source, est_source, name='pit_snr'):
        """
        :param source: [B, nspk, K, L]
        :param est_source: [B, nspk, K, L]
        :param name:
        :return:
        """
        max_len = tf.shape(source)[2]   #
        # mask the padding part and flat the segmentation
        # zero-mean source and recon in the real length
        # seq_mask = self.get_seq_mask(max_len, self.K)
        # seq_mask = tf.reshape(seq_mask, [self.batch_size, 1, -1, 1])
        # mask_targets = source * seq_mask
        # mask_recon = est_source * seq_mask
        sample_count = tf.cast(tf.reshape(self.batch_size * [self.K * L], [self.batch_size, 1, 1, 1]), tf.float32)
        mean_targets = tf.reduce_sum(source, axis=[2, 3], keepdims=True) / sample_count
        mean_recon = tf.reduce_sum(est_source, axis=[2, 3], keepdims=True) / sample_count
        zero_mean_targets = source - mean_targets
        zero_mean_recon = est_source - mean_recon
        # shape is [B, nspk, s]
        flat_targets = tf.reshape(zero_mean_targets, [self.batch_size, self.nspk, -1])
        flat_recon = tf.reshape(zero_mean_recon, [self.batch_size, self.nspk, -1])

        # calculate the SI-SNR, PIT is necessary
        with tf.variable_scope(name):
            v_perms = tf.constant(
                list(permutations(range(self.nspk))),
                dtype=tf.int32)
            perms_one_hot = tf.one_hot(v_perms, depth=self.nspk, dtype=tf.float32)

            # shape is [B, 1, nspk, s]
            s_truth = tf.expand_dims(flat_targets, axis=1)
            # shape is [B, nspk, 1, s]
            s_estimate = tf.expand_dims(flat_recon, axis=2)
            pair_wise_dot = tf.reduce_sum(s_estimate * s_truth, axis=3, keepdims=True)
            s_truth_energy = tf.reduce_sum(s_truth ** 2, axis=3, keepdims=True) + self.eps
            pair_wise_proj = pair_wise_dot * s_truth / s_truth_energy
            e_noise = s_estimate - pair_wise_proj
            # shape is [B, nspk, nspk]
            pair_wise_snr = tf.div(tf.reduce_sum(pair_wise_proj ** 2, axis=3),
                                   tf.reduce_sum(e_noise ** 2, axis=3) + self.eps)
            pair_wise_snr = 10 * tf.log(pair_wise_snr + self.eps) / tf.log(10.0)  # log operation use 10 as base
            snr_set = tf.einsum('bij,pij->bp', pair_wise_snr, perms_one_hot)
            max_snr_idx = tf.cast(tf.argmax(snr_set, axis=1), dtype=tf.int32)
            max_snr = tf.gather_nd(snr_set,
                tf.stack([tf.range(self.batch_size, dtype=tf.int32), max_snr_idx], axis=1))
            max_snr = max_snr / self.nspk

        return max_snr, v_perms, max_snr_idx

import tensorflow as tf

def bilstm_embedding(input_data,  wvs, **kwargs):
    with tf.variable_scope('bilstm'):
        wv_params = tf.Variable(wvs, name='word2vec', dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(wv_params, input_data, None, name='inputs')
        state_size = kwargs.get('state_size', 128)
        lstm_num_layers = kwargs.get('lstm_num_layers', 1)
        num_outputs = kwargs.get('num_outputs', 128)
        cells_fw = [tf.nn.rnn_cell.BasicLSTMCell(state_size) for _ in range(lstm_num_layers)]
        cells_bw = [tf.nn.rnn_cell.BasicLSTMCell(state_size) for _ in range(lstm_num_layers)]

        initial_states_fw = [cell.zero_state(input_data.shape[0], tf.float32) for cell in cells_fw]
        initial_states_bw = [cell.zero_state(input_data.shape[0], tf.float32) for cell in cells_bw]

        rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(inputs, inputs.shape[1], axis=1)]
        rnn_outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=rnn_inputs,
                                                           initial_states_fw=initial_states_fw,
                                                           initial_states_bw=initial_states_bw)
        rnn_outputs = [output_states_fw[0].c, output_states_fw[0].h, output_states_bw[0].c, output_states_bw[0].h]
        rnn_outputs = tf.concat(rnn_outputs, axis=1, name='rnn_outputs')
        outputs = tf.contrib.layers.fully_connected(rnn_outputs, num_outputs = num_outputs, scope='outputs')
    return outputs
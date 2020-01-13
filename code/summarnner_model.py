from mxnet import gluon, nd
from mxnet.gluon import nn


class SentenceRecurrent(nn.Block):
    """Eqn. (6)

    """

    def __init__(self, sentence_hidden_size, ** kwargs):
        super(SentenceRecurrent, self).__init__(**kwargs)
        # content
        self.content_encoder = nn.Dense(1, flatten=False, use_bias=True)
        # salience
        self.salience_encoder = nn.Dense(sentence_hidden_size * 2, flatten=False, use_bias=False)
        # novelty
        self.novelty_encoder = nn.Dense(sentence_hidden_size * 2, flatten=False, use_bias=False)

        # pos
        # self.abs_pos_encoder = nn.Dense(1, flatten=False)
        # self.rel_pos_encoder = nn.Dense(1, flatten=False)

    def forward(self, current, previous, doc_encode):
        """[summary]

        Args:
            current ([type]): h_j (batch_size, sentence_hidden_size * 2)
            previous ([type]): s_j (batch_size, sentence_hidden_size * 2)
            doc_encode ([type]): d (batch_size, ndoc_dims)
        """
        # content: (batch_size, 1)
        content = self.content_encoder(current)
        # salience: (batch_size, sentence_hidden_size * 2)
        salience = self.salience_encoder(doc_encode)
        salience = current * salience
        # salience: (batch_size,)
        salience = nd.sum_axis(salience, -1)
        # salience: (batch_size, 1)
        salience = nd.expand_dims(salience, -1)

        # novelty: (bathc_size, sentence_hidden_size * 2)
        novelty = self.novelty_encoder(nd.tanh(previous))
        novelty = current * novelty
        # salience: (batch_size,)
        novelty = nd.sum_axis(novelty, -1)
        # salience: (batch_size, 1)
        novelty = nd.expand_dims(novelty, -1)

        # P: (batch_size, 1)
        P = nd.sigmoid(content + salience - novelty)

        return P


class Encoder(nn.Block):
    def __init__(
            self, nwords, nword_dims, word_hidden_size, word_nlayers, sentence_hidden_size,
            sentence_nlayers, ndoc_dims, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        with self.name_scope():
            self.embedding = nn.Embedding(nwords, nword_dims)
            self.word_rnn = gluon.rnn.LSTM(
                word_hidden_size, num_layers=word_nlayers, layout='NTC', bidirectional=True)
            self.sentence_rnn = gluon.rnn.LSTM(
                sentence_hidden_size, num_layers=sentence_nlayers, layout='NTC', bidirectional=True)
            self.fully_encoder = nn.Dense(ndoc_dims, activation='tanh', flatten=False)

    def forward(self, inputs):
        """
        Args:
            inputs (NDArray): (batch_size, n_sentences, sentence_length)

        Returns:
            NDArray: [description]
        """
        # embed: (batch_size, n_sentences, sentence_length, nword_dims)
        embed = self.embedding(inputs)
        # (batch_size, n_sentences, sentence_length, word_hidden_size)
        word_hidden = nd.stack(*[self.word_rnn(e) for e in embed])
        # (batch_size, n_sentences, word_hidden_size * 2)
        word_hidden = nd.mean(word_hidden, axis=2)
        # (batch_size, n_sentences, sentence_hidden_size * 2)
        sentence_hidden = self.sentence_rnn(word_hidden)
        # doc representation (batch_size, sentence_hidden_size * 2)
        doc_encode = nd.mean(sentence_hidden, axis=1)
        # doc representation (batch_size, ndense_units)
        doc_encode = self.fully_encoder(doc_encode)

        return sentence_hidden, doc_encode


if __name__ == "__main__":

    # %%
    encoder = Encoder(1000, 300, 200, 2, 100, 2, 500,)
    encoder.initialize()
    rnn = SentenceRecurrent(100)
    rnn.initialize()

    # %%
    X = nd.ones((64, 20, 10))
    sentence_hidden, doc_encode = encoder(X)
    sentence_hidden.shape, doc_encode.shape

    # %%
    p = rnn(nd.ones((64, 200)), nd.ones((64, 200)), doc_encode)
    print(p.shape)

    # %%

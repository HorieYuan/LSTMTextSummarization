import gc
import time
import os
import gluonnlp as nlp
import mxnet as mx
from mxnet import autograd, gluon, init, nd, test_utils
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss


def batch_loss(encoder, sent_rnn, X, Y, vocab, loss, ctx):

    batch_size = X.shape[1]
    sentence_hidden, doc_encode = encoder(X)
    sentence_hidden = nd.transpose(sentence_hidden, axes=(1, 0, 2))

    # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
    # mask, num_not_pad_tokens = nd.ones(shape=(batch_size,), ctx=ctx), 0
    l = nd.array([0], ctx=ctx)

    # 以前所有步
    previous = sentence_hidden[0]

    # sent_hidden: (batch_size, hidden)
    for sent_hidden, y in zip(sentence_hidden, Y.T):

        y_h = sent_rnn(sent_hidden, previous, doc_encode)
        y_h = nd.squeeze(y_h)

        los = loss(y_h, y).sum()
        # print('los', los)
        l = l + los

        # 公式 7，这里使用强制教学
        y = nd.expand_dims(y, -1)
        previous = previous + sent_hidden * y

    return l / batch_size


def train(encoder, sent_rnn, dataset, lr, batch_size, num_epochs, vocab, ctx):

    print('start training')

    encoder.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    sent_rnn.initialize(init.Xavier(), force_reinit=True, ctx=ctx)

    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam', {'learning_rate': lr})
    sent_rnn_trainer = gluon.Trainer(sent_rnn.collect_params(), 'adam', {'learning_rate': lr})

    loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    best_batch = 0
    best_loss = float('Inf')

    for epoch in range(num_epochs):
        l_sum = 0.0
        for i, (X, Y) in enumerate(data_iter):

            X = X.as_in_context(ctx)
            Y = Y.as_in_context(ctx)

            with autograd.record():
                l = batch_loss(encoder, sent_rnn, X, Y, vocab, loss, ctx)
            l.backward()
            sent_rnn_trainer.step(X.shape[1])
            enc_trainer.step(X.shape[1])
            l_sum += l.asscalar()

            if i % 20 == 0:
                info = "epoch %d, batch %d, batch_loss %.3f" % (epoch, i, l.asscalar())
                print(info)

        cur_loss = l_sum / len(data_iter)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_batch = epoch + 1

            if not os.path.exists('../model'):
                os.mkdir('../model')
            encoder.save_parameters('../model/encoder' + str(epoch + 1) + '.params')
            sent_rnn.save_parameters('../model/sent_rnn' + str(epoch + 1) + '.params')

        info = "epoch %d, loss %.3f, best_loss %.3f, best_batch %d" % (
            epoch, cur_loss, best_loss, best_batch)
        print(info)

        # log
        if not os.path.exists('../log'):
            os.mkdir('../log')
        with open('../log/log.log', 'a', encoding='utf-8') as fa:
            fa.write(time.ctime() + "\t" + info + '\n')

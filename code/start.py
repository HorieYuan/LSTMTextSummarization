import json
import pickle

from flask import Flask, jsonify, request

import hyper_parameters as hp
from summarize import summarize
from summarnner_model import Encoder, SentenceRecurrent
from utils import _prepare_predict_data, prepare_predict_data, try_gpu


N_EPOCHS = hp.N_EPOCHS
BATCH_SIZE = hp.BATCH_SIZE

EMBED_SIZE = hp.EMBED_SIZE

WORD_HIDDEN_SIZE = hp.WORD_HIDDEN_SIZE
WORD_NLAYERS = hp.WORD_NLAYERS

SENTENCE_HIDDEN_SIZE = hp.SENTENCE_HIDDEN_SIZE
SENTENCE_NLAYERS = hp.SENTENCE_NLAYERS

NDOC_DIMS = hp.NDOC_DIMS
LR = hp.LR

VOCAB_PATH = hp.VOCAB_PATH
SENT_RNN_MODEL_PATH = hp.SENT_RNN_MODEL_PATH
ENCODER_MODEL_PATH = hp.ENCODER_MODEL_PATH
CTX = try_gpu()

PORT = hp.PORT

app = Flask(__name__)

word_vocab = None
encoder = None
sent_rnn = None


@app.route('/')
def do_summarize():
    source = ''
    src_type = ''
    json_result = {"status": "success"}

    try:
        if request.method == 'GET':
            source = request.args.get('source', default='')
            src_type = request.args.get('src_type', default='json')
        else:
            source = request.form.get('source', default='')
            src_type = request.form.get('src_type', default='json')

        if src_type == 'json':
            source = json.loads(source)
            source = ''.join(source['paragraphs'])

        result = summarize(encoder, sent_rnn, word_vocab, source)

        if src_type == 'json':
            summary = {}
            for idx, sent in enumerate(result):
                summary['sentence' + str(idx)] = sent
            json_result['summary'] = summary
            result = json.dumps(json_result, ensure_ascii=False)
        else:
            result = ''.join(result)
    except Exception as e:
        print(e)
        json_result["error_mag"] = str(e)
        json_result["status"] = "failed"
        result = json.dumps(json_result, ensure_ascii=False)

    return result


if __name__ == '__main__':

    with open(VOCAB_PATH, 'rb') as fr:
        word_vocab = pickle.load(fr)
    NWORDS = len(word_vocab)
    encoder = Encoder(NWORDS, EMBED_SIZE, WORD_HIDDEN_SIZE, WORD_NLAYERS,
                      SENTENCE_HIDDEN_SIZE, SENTENCE_NLAYERS, NDOC_DIMS)
    sent_rnn = SentenceRecurrent(SENTENCE_HIDDEN_SIZE)
    encoder.load_parameters(ENCODER_MODEL_PATH, ctx=CTX)
    sent_rnn.load_parameters(SENT_RNN_MODEL_PATH, ctx=CTX)

    
    app.run(debug=True, host="0.0.0.0", port=PORT)

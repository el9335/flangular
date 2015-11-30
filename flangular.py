from flask import Flask, request, jsonify, current_app, make_response, send_file
import nltk
import os
from nltk import Tree
import json
import datetime
from functools import update_wrapper
from datetime import timedelta
from stat_parser import Parser
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters



########################
# load resource models #
########################



punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

parser = Parser()

app = Flask(__name__)

########################
# https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
########################

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

########################
# NLP Resources        #
########################

# tokenize input text for app visualization
@app.route('/flangular-nlp/v3.0/services/viz/tokens',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def tokenize():

	return jsonify(results=nltk.word_tokenize(request.args['text']))

# segmentize text for app visualization
@app.route('/flangular-nlp/v3.0/services/viz/sentences',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def segmentize():

	return jsonify(results=sentence_splitter.
				tokenize(request.args['text'].replace('?"', '? "').
					replace('!"', '! "').replace('."', '. "')))

# apply part of speech tagging to text for app visualization
@app.route('/flangular-nlp/v3.0/services/viz/tagged-tokens',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def tag():

	return jsonify(results=nltk.pos_tag(nltk.word_tokenize(request.args['text'])))

# identify named entities for app visualization
@app.route('/flangular-nlp/v3.0/services/viz/entities',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def chunk():
	
	return jsonify(results=nltk.chunk.ne_chunk
			(nltk.pos_tag(nltk.word_tokenize(request.args['text']))))

# build dependency tree for app visualization
@app.route('/flangular-nlp/v3.0/services/viz/parsetree',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def deptree():

    sentences = sentence_splitter.tokenize(request.args['text'].replace('?"', '? "').
                     replace('!"', '! "').replace('."', '. "'))

    sents = []
    for s in sentences:
        sents.append(str(parser.parse(s)))

    return(jsonify(results=sents))

    # results=[]
    # for sentence in sentences:
    #     results.append(str(parser.parse(sentence)))

    #return jsonify(results)
    #if num segments greater than 1, return error.  here and for download..

    # now = datetime.datetime.now()
    # filename = "fnlp_tree_img--" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second)

    # t = str(parser.parse(request.args['text'])).strip('\n')
    # tree = Tree.fromstring(t)

    # tree.draw()

    # cf = CanvasFrame()
    # tc = TreeWidget(cf.canvas(),tree)

    # cf.add_widget(tc,10,10)
    # cf.print_to_file(filename + ".ps")

    # command = "convert " + filename + ".ps " + filename + ".png"  

    # os.system(command)

    # img_location = os.getcwd() + os.sep + filename + "/png" 

    #needs to be universal
    #return send_file(filename + ".png", mimetype='image/gif')
    #return 1
    
	# return jsonify(results=str(parser.parse(request.args['text'])).strip('\n'))

# tokenize input text for download
@app.route('/flangular-nlp/v3.0/services/download/tokens',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def tokenizeDL():

    path = "."+ os.sep + "tokens"
    if not os.path.isdir(path):
        os.mkdir(path)

    now = datetime.datetime.now()
    out_filename = path + os.sep + "fnlp_tokens" + "_" + str(now.year) + "-" + str(now.day) +"-" + str(now.month) + "--" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second) + ".txt"

    results=nltk.word_tokenize(request.args['text'])
    try:
        with open(out_filename,'w') as outfile:
            json.dump(results, outfile)

        return str(os.getcwd() + out_filename.strip('.'))
    except:
        return '1'


# segmentize text for app visualization for download
@app.route('/flangular-nlp/v3.0/services/download/sentences',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def segmentizeDL():

    path = "."+ os.sep + "sentences"
    if not os.path.isdir(path):
        os.mkdir(path)

    now = datetime.datetime.now()
    out_filename = path + os.sep + "fnlp_sentences" + "_" + str(now.year) + "-" + str(now.day) +"-" + str(now.month) + "--" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second) + ".txt"

    results = sentence_splitter.tokenize(request.args['text'].replace('?"', '? "').
                    replace('!"', '! "').replace('."', '. "'))


    #response = make_response(json.dump(results))
    #response.headers["Content-Disposition"] = "attachment; filename=sentences.csv"
    
    #return response
    try:
        with open(out_filename,'w') as outfile:
            json.dump(results, outfile)

        return str(os.getcwd() + out_filename.strip('.'))
    except:
        return '1'

# apply part of speech tagging to text for download
@app.route('/flangular-nlp/v3.0/services/download/tagged-tokens',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def tagDL():

    path = "."+ os.sep + "tags"
    if not os.path.isdir(path):
        os.mkdir(path)

    now = datetime.datetime.now()
    out_filename = path + os.sep +"fnlp_tags" + "_" + str(now.year) + "-" + str(now.day) +"-" + str(now.month) + "--" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second) + ".txt"

    results = nltk.pos_tag(nltk.word_tokenize(request.args['text']))

    try:
        with open(out_filename,'w') as outfile:
            json.dump(results, outfile)

        return str(os.getcwd() + out_filename.strip('.'))
    except:
        return '1'

# identify named entities for download
@app.route('/flangular-nlp/v3.0/services/download/entities',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def chunkDL():

    path = "."+ os.sep + "entities"
    if not os.path.isdir(path):
        os.mkdir(path)

    now = datetime.datetime.now()
    out_filename = path + os.sep + "fnlp_entities" + "_" + str(now.year) + "-" + str(now.day) +"-" + str(now.month) + "--" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second) + ".txt"

    results = nltk.chunk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(request.args['text'])))

    try:
        with open(out_filename,'w') as outfile:
            json.dump(results, outfile)

        return str(os.getcwd() + out_filename.strip('.'))
    except:
        return '1'

# build dependency tree for download - currently supports one tree per request
@app.route('/flangular-nlp/v3.0/services/download/parsetree',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def deptreeDL():

    sentence = sentence_splitter.tokenize(request.args['text'].replace('?"', '? "').
                    replace('!"', '! "').replace('."', '. "'))

    path = "."+ os.sep + "trees"
    if not os.path.isdir(path):
        os.mkdir(path)

    now = datetime.datetime.now()
    out_filename = path + os.sep +"fnlp_tree" + "_" + str(now.year) + "-" + str(now.day) +"-" + str(now.month) + "--" + str(now.hour) + "-" + str(now.minute) + "-" + str(now.second) + ".txt"

    sentences = sentence_splitter.tokenize(request.args['text'].replace('?"', '? "').
                     replace('!"', '! "').replace('."', '. "'))

    sents = []
    for s in sentences:
        sents.append(str(parser.parse(s)).strip('\n'))

    try:
        with open(out_filename,'w') as outfile:
            json.dump(sents, outfile)

        return str(os.getcwd() + out_filename.strip('.'))
    except:
        return '1'

if __name__ == "__main__":
    app.run(debug=True)

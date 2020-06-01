from bot import *
import random
import tensorflow as tf
import tflearn
from flask import Flask, request, session, flash, json, render_template,jsonify

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load('model.tflearn')

app = Flask(__name__)
app.secret_key = 'chat bot'

with open("intents.json") as file:
    data = json.load(file)
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

def bot_ai(msg):
    results = model.predict([bag_of_words(msg, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    return random.choice(responses)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/process',methods=['POST'])
def process():
    msg = request.form.get('msg')
    if msg:
        result  = bot_ai(msg)
        return jsonify({'bot':result})
    else:
        return jsonify({'bot':"dont give me that silent treatment"})
if __name__ == "__main__":
    app.run(debug=True)
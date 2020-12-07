from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from Model_translater import Model_translater
import pickle
import sqlite3
import os
import numpy as np


app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
translater = Model_translater()


class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired()])
                                # ,validators.length(min=10)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        content = request.form['moviereview']
        EN, VN = translater.translate(content)
        return render_template('reviewform.html',form=form,
                                content=VN)
    return render_template('reviewform.html', form=form)


if __name__ == '__main__':
#     app.run(debug=True)
    app.run(port = '8080',host = '0.0.0.0', debug=True)

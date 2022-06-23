from flask import *
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    my_prediction=model.predict(final)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)

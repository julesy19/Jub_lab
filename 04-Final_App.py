from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model,scaler,sample_json):

    # Pour les features de données plus volumineuses,
    # vous devriez probablement écrire une boucle for
    # Cela construit ce tableau pour vous

    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len,s_wid,p_len,p_wid]]

    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])

    class_ind = np.argmax(model.predict(flower), axis=-1)

    return classes[class_ind][0]



app = Flask(__name__)
# Configurer une clé secrète SECRET_KEY
# Nous apprendrons plus tard de bien meilleures façons de le faire !!
app.config['SECRET_KEY'] = 'mysecretkey'


# N'OUBLIEZ PAS DE CHARGER LE MODÈLE ET LE SCALER !
flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")


# Maintenant, créez une classe WTForm
# Beaucoup de champs disponibles :
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    sep_len = TextField('Longueur du sépale ')
    sep_wid = TextField('Largeur du sépale ')
    pet_len = TextField('Longueur du pétale ')
    pet_wid = TextField('Largeur du pétale ')

    submit = SubmitField('Analyser')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Créer une instance du formulaire.
    form = FlowerForm()
    # Si le formulaire est valide lors de la soumission
    # (nous parlerons ensuite de validation)
    if form.validate_on_submit():
        # Récupérer les données de l'espèce sur le formulaire

        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])

    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)

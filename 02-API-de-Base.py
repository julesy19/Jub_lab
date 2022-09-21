from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib


#### C'EST CE QUE NOUS FAISONS DANS POSTMAN ###
# ÉTAPE 1 : Créer une nouvelle requête
# ÉTAPE 2 : Sélectionnez POST
# ÉTAPE 3 : Tapez l'URL correcte (http://127.0.0.1:5000/api/flower)
# ÉTAPE 4 : Sélectionner le Body
# ÉTAPE 5 : Sélectionner JSON
# ÉTAPE 6 : Tapez ou Coller pour l'exemple une requête json
# ÉTAPE 7 : Exécutez 02-API-de-Base.py pour lancer le serveur et confirmer que le site est en cours d'exécution
# Étape 8 : Exécuter une demande d'API

def return_prediction(model,scaler,sample_json):

    # Pour les features de données plus volumineuses, vous devriez probablement écrire une boucle for
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

@app.route('/')
def index():
    return "<h1>L'APPLICATION FLASK EST EN COURS D'EXÉCUTION !</h1>"


# N'OUBLIEZ PAS DE CHARGER LE MODÈLE ET LE SCALER !
flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

@app.route('/api/flower', methods=['POST'])
def predict_flower():

    content = request.json

    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
    return jsonify(results)

if __name__ == '__main__':
    app.run()

import keras.optimizers.schedules
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import pandas as pd

#Bart - 0
#Homer - 1
nome_classes = ['Bart', 'Homer']
epochs = 100
batch_size = 2

#Carrega e trata base de dados separando previsores e classe
def CarregaDados():

    dados = pd.read_csv('personagens.csv')

    previsores = dados.drop(['classe'], axis=1)
    classe = pd.DataFrame()
    classe.insert(loc=0, column='classe', value=dados['classe'])

    #encoder = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ['classe']), remainder='passthrough', sparse_threshold=False)
    encoder = LabelEncoder()
    classe['classe'] = encoder.fit_transform(classe)

    return previsores, classe

#Cria rede neural classificadore com duas camadas ocultas e dropout
def CriaRede():
    modelo = Sequential()

    modelo.add(Dense(units=20, activation='relu', input_dim=6))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=20, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=1, activation='sigmoid'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.006,
        decay_steps=13200,
        decay_rate=0.0006
    )

    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler), loss='binary_crossentropy', metrics='accuracy')

    return modelo

#Treina e salva modelo com resultado grafico
def GeraModelo():
    previsores, classe = CarregaDados()

    modelo = CriaRede()

    result = modelo.fit(previsores, classe, batch_size=batch_size, epochs=epochs)

    modelo.save('Modelo.0.1')

    plt.plot(result.history['loss'])
    plt.title('Relação Função de Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()

    plt.plot(result.history['accuracy'])
    plt.title('Relação de Acuracia')
    plt.xlabel('Épocas')
    plt.ylabel('Taxa de Acerto')
    plt.show()


def Previsao(previsores):
    modelo = load_model('Modelo.0.1')

    result = modelo.predict(previsores)

    if result > 0.5:
        return 'Homer'
    else:
        return 'Bart'

#Realiza treinamento com validação cruzada e analise de desvio padrão
def Treinamento():

    previsores, classe = CarregaDados()

    modelo = KerasClassifier(build_fn=CriaRede, epochs=epochs, batch_size=batch_size)

    result = cross_val_score(estimator=modelo, X=previsores, y=classe, cv=10)

    print('Resultado:'+str(result))
    print('Média:'+str(result.mean()))
    print('Desvio Padrão:'+str(result.std()))


Treinamento()

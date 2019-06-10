import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.callbacks import EarlyStopping

def create_neural_net(input_dim, n_hid, learn_rate, dropout):
    model = Sequential()
    model.add(Dense(n_hid[0], activation='relu', kernel_initializer='he_uniform',input_dim=input_dim))
    model.add(Dropout(dropout))
    for i in range(1, len(n_hid)):
        model.add(Dense(n_hid[i], activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(dropout))
    model.add(Dense(48, activation='softmax', kernel_initializer='he_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def read_data():
    file = pd.read_csv('../embedding_csv/english.csv')
    file = file.sample(frac=1)
    features = file.drop(['index', 'intent'], axis=1).values
    intents = file['intent'].values
    data_points, num_features = features.shape
    return features, intents, data_points, num_features, file

def train_neural_network_seperate(features, intents, input_dim, num_intents, n_hid, learn_rate, num_epochs, cb_list, dropout):
    num_splits=10
    skf = StratifiedKFold(n_splits=num_splits, random_state=1)
    split_records = []
    models = []
    model_history = []
    split = 1
    for train_index, test_index in skf.split(features, intents):
        mlp = create_neural_net(input_dim, n_hid, learn_rate, dropout)
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = intents[train_index], intents[test_index]
        y_train = to_categorical(y_train, num_classes=num_intents, dtype='float32')
        y_test = to_categorical(y_test, num_classes=num_intents, dtype='float32')
        history = mlp.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=num_epochs, verbose=2, shuffle=True, callbacks=cb_list)
        models.append(mlp)
        model_history.append(history)
        train_predictions = mlp.predict(x_train)
        train_predictions = (train_predictions>0.5)
        test_predictions = mlp.predict(x_test)
        test_predictions = (test_predictions>0.5)
        train_accuracy = float('{0:.3f}'.format(accuracy_score(y_train, train_predictions)))
        test_accuracy = float('{0:.3f}'.format(accuracy_score(y_test, test_predictions)))
        difference = float('{0:.3f}'.format(train_accuracy-test_accuracy))
        print('train_accuracy = {}, test_accuracy = {}'.format(train_accuracy, test_accuracy))
        temp = [split, n_hid, train_accuracy, test_accuracy, difference, dropout, learn_rate]
        split_records.append(temp)
        split+=1
    return split_records, models, model_history

def store_records(model_history, num_hidden, train_iteration, split_records):
    pp = PdfPages('../plots/mlp/model_accuracy_plot_'+str(train_iteration)+'_'+str(num_hidden)+'.pdf')
    for history in model_history:
        fig = plt.figure()
        plt.plot(history.history['categorical_accuracy'], 'r-')
        plt.plot(history.history['val_categorical_accuracy'], 'b-')
        plt.title('model_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        pp.savefig(fig)
    pp.close()
    pp = PdfPages('../plots/mlp/model_loss_plot_'+str(train_iteration)+'_'+str(num_hidden)+'.pdf')
    for history in model_history:
        fig = plt.figure()
        val_loss = history.history['val_loss']
        loss = history.history['loss']
        plt.plot(loss, 'r-')
        plt.plot(val_loss, 'b-')
        plt.title('model_loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        pp.savefig(fig)
    pp.close()
    df = pd.DataFrame(split_records)
    with open('../training_records/mlp/multiple_model_train_records.csv', 'a') as f:
        df.to_csv(f, header = ['iteration', 'hidden_nodes', 'train_accuracy', 'test_accuracy', 'difference', 'dropout', 'learning_rate'], index_label='index')

def get_best_model(split_records):
    records = []
    for record in split_records:
        if record[4]>0:
            records.append(record)
    df = pd.DataFrame(records, columns=['index', 'hidden_nodes' ,'train_accuracy', 'test_accuracy', 'difference', 'dropout', 'learning_rate'])
    df = df.sort_values(by=['difference'])
    return df

def read_test_data(language):
    file = pd.read_csv('../embedding_csv/'+language+'.csv')
    features = file.drop(['index', 'intent'], axis=1).values
    intents = file['intent']
    return features, intents

def get_predictions(model, features):
    predictions = model.predict(features)
    intent_predicted = np.argmax(predictions, axis=1)
    return intent_predicted

def get_accuracy(model, features, intents):
    intent_predicted = get_predictions(model, features)
    accuracy = accuracy_score(intents, intent_predicted)
    return accuracy

def evaluate(mlp):
    languages = ['hindi', 'arabic', 'spanish']
    for language in languages:
        features, intents = read_test_data(language)
        accuracy = get_accuracy(mlp, features, intents)
        print(language, accuracy)

features, intents, data_points, num_features, dataset = read_data()
num_intents = len(set(intents))
print(num_intents)
print(set(intents))
n_hid = [100, 100, 100]
learn_rate = 0.1
dropout = 0.2
num_epochs = 180
es = EarlyStopping(monitor='categorical_accuracy', min_delta=0.00001, patience=5, restore_best_weights=True, verbose=1, baseline=0.01)
cb_list = [es]

train_iteration= 0

train_iteration += 1
split_reocrds, models, model_history = train_neural_network_seperate(features, intents, num_features, num_intents, n_hid, learn_rate, num_epochs, cb_list, dropout)

store_records(model_history, n_hid, train_iteration, split_reocrds)

sorted_stats = get_best_model(split_reocrds)

print(sorted_stats)

mlp = models[1]

mlp.summary()

evaluate(mlp)
from metrics import *


def evaluate_model(model, X_train, y_train, X_val, y_val):
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    prec_train = precision(y_train, pred_train)
    prec_val = precision(y_val, pred_val)
    recall_train = recall(y_train, pred_train)
    recall_val = recall(y_val, pred_val)
    f1_train = fbeta_score(y_train, pred_train)
    f1_val = fbeta_score(y_val, pred_val)
    val_results = "Validation Results: Prec - {:.2f}  Recall - {:.2f}  F1 - {:.2f}".format(prec_val,
                                                                                        recall_val,
                                                                                        f1_val)
    train_results = "Train Results:      Prec - {:.2f}  Recall - {:.2f}  F1 - {:.2f}".format(prec_train,
                                                                                          recall_train,
                                                                                          f1_train)
    print("{}\n{}".format(train_results, val_results))
    return


def create_embedding_matrix(filepath, dictionary, embedding_dim):
    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(dictionary)+1, embedding_dim))
    for word, i in dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.regularizers import l2

def plot_training_history(history):
    plt.figure(figsize=(3.3, 6))
    epoch_range = list(range(1, len(history.history['accuracy']) + 1))
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    sub_title = f"Lowest validation loss at epoch: {best_epoch}"

    plt.subplot(2, 1, 1)
    plt.plot(epoch_range, history.history['accuracy'], label='Train', color='royalblue', linestyle='-')
    plt.plot(epoch_range, history.history['val_accuracy'], label='Validation', color='#008080', linestyle='-')
    plt.title('Accuracy', weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axvline(x=best_epoch, color='red', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.plot(epoch_range, history.history['loss'], label='Training', color='royalblue', linestyle='-')
    plt.plot(epoch_range, history.history['val_loss'], label='Validation', color='#008080', linestyle='-')
    plt.title('Loss', weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.axvline(x=best_epoch, color='red', linestyle='--')

    plt.subplots_adjust(hspace=0.38)

    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='upper center', ncol=3)
    plt.savefig("Training_history.pdf", bbox_inches='tight')
    print(sub_title)
    plt.show()
    
def extract_embedding(column, max_length):
    model_name = "prajjwal1/bert-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)

    tokenized_input = tokenizer(column.values.tolist(),padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
      hidden = bert_model(**tokenized_input)
    return hidden.last_hidden_state

def build_model(n_LSTM_ev, n_LSTM_cl, n_DENSE1, n_DENSE2, DROPOUT_RATE1, DROPOUT_RATE2, L2_REG, RET_SEQ_ev, RET_SEQ_cl, DROPOUT_rec):
    ##### Inputs #####
    # BERT embeddings of the claims
    claim_input = Input(shape=(27, 512), name="Claim")
    # BERT embeddings for the evidence
    evidence_input = Input(shape=(127, 512), name="Evidence")
    
    #### Bidirectional LSTM layers ####
    # For the evidence
    biLSTM_evidence = Bidirectional(LSTM(units=n_LSTM_ev, return_sequences=RET_SEQ_ev, 
                                        recurrent_dropout=DROPOUT_rec,
                                         kernel_regularizer=l2(L2_REG),
                                         recurrent_regularizer=l2(L2_REG),
                                         activity_regularizer=l2(L2_REG),
                                         name="LSTM1"),
                                    name="BiLSTM-evidence")(evidence_input)
    # For the claim
    biLSTM_claim = Bidirectional(LSTM(units=n_LSTM_cl, return_sequences=RET_SEQ_cl, 
                                     recurrent_dropout=DROPOUT_rec,
                                      kernel_regularizer=l2(L2_REG),
                                      recurrent_regularizer=l2(L2_REG),
                                      activity_regularizer=l2(L2_REG),
                                      name="LSTM2"),
                                 name="BiLSTM-claim")(claim_input)
    
    #### Concatenate the two LSTM layers ####
    concat = Concatenate(name="Concat", axis=1)([biLSTM_evidence, biLSTM_claim])
    
    #### Dense block 1 ####
    # Dropout
    dropout_layer1 = Dropout(DROPOUT_RATE1, name="Dropout1")(concat)
    
    # Dense
    dense_layer1 = Dense(units=n_DENSE1,
                         kernel_regularizer=l2(L2_REG),
                         activity_regularizer=l2(L2_REG),
                         activation='relu', name="Dense1")(dropout_layer1)
    # Batch normalization
    BN1 = BatchNormalization(name="BN1")(dense_layer1)
    
    #### Dense block 2 ####
    # Dropout
    dropout_layer2 = Dropout(DROPOUT_RATE2, name="Dropout2")(BN1)
    
    # Dense
    dense_layer2 = Dense(units=n_DENSE2,
                         kernel_regularizer=l2(L2_REG),
                         activity_regularizer=l2(L2_REG),
                         activation='relu', name="Dense2")(dropout_layer2)
    
    # Batch normalization
    BN2 = BatchNormalization(name="BN2")(dense_layer2)
    
    #### Output #####
    output = Dense(units=1, activation='sigmoid', name="Output")(BN2)

    model = Model([claim_input, evidence_input], output)
    
    return model

def fit_model(model, LR, BS, claim_train, evidence_train, y_train, claim_val, evidence_val, y_val):
    """
    Fits a model with leanring rate LR and batch size BS.
    """
    model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("FactVerModel", 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=1)
    
    history = model.fit([claim_train, evidence_train],
                        y_train, epochs=200, batch_size=BS,
                        validation_data=([claim_val, evidence_val], y_val),
                       callbacks=[es, checkpoint], verbose=1)
    best_model = load_model("FactVerModel")
    return history, best_model
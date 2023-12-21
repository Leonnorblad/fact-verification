# Functions to evaluete the model
from transformers import AutoTokenizer
from fact_ver import extract_embedding
import pandas as pd
from tensorflow.keras.models import Model, load_model
model = load_model("FactVerModel")

def count_tokens(text):
    """
    Takes a text and returns the number of tokens (ChatGPT 3.5-turo tokenizer)
    """
    model_name = "prajjwal1/bert-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_input = tokenizer(text, return_tensors="pt")
    num_tokens = tokenized_input["input_ids"].shape[1]
    return num_tokens

def check_tunkated(claim, evidence):
    """
    Return True if the cliam or evidence will be truncated in the embedding 
    """
    num_cl = count_tokens(claim)
    num_ev = count_tokens(evidence)
    if num_cl>27 or num_ev>127:
        return True
    else:
        return False
    
def mod_pred(claim, evidence):
    """
    Takes two strings, claim and evidence and returns the model prediction.
    """
    # Check input lengths
    if check_tunkated(claim, evidence):
        print("Warning! Input will be truncated by the embedder")
        
    # Prepare input
    df = pd.DataFrame({'cl':[claim], 'ev':[evidence]})
    evidence_embedding = extract_embedding(df['ev'], 127).numpy()
    claim_embedding = extract_embedding(df['cl'], 27).numpy()
    
    # Use model to get prediction
    model_prediction = model.predict([claim_embedding, evidence_embedding], verbose=0)
    
    return model_prediction[0][0]
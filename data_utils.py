import re
import config
import tensorflow as tf


################################Utility functions####################

def decontracted(phrase):

    """
    This funtion is for preprocssing the given phrase.
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase.lower()


def read_lines(file_path):

    with open(file_path,"r") as f:
        train_text = f.readlines()
    return train_text



def get_data(file_path):
    
    train_text = read_lines(file_path)
    
    train = []    
    for sent in train_text[1:]:
        if sent.split(" ")[0] != "\n":
            train.append(sent.split(" ")[0])
        else:
            train.append("</s>")

    train_sent = " ".join(train)
    
    train_data = []
    for es in train_sent.split("</s>"):
        es = es.strip()
        if len(es) > 0:
            es = "<s> " + es + " </s>"
            train_data.append(es)
    
    return [decontracted(phrase) for phrase in train_data]

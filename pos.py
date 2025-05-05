import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import classification_report
from hmmlearn.hmm import CategoricalHMM                    
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

# Pre-processing and setUp
def read_text_file(filepath):
    sentences = []
    sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            token, morph, pos = parts
            sentence.append((token, pos))

            if pos == 'PUNC' and token in ['.', '?', '!']:
                if sentence:
                    sentences.append(sentence)
                    sentence = []

        if sentence:
            sentences.append(sentence)

    return sentences  


# - Split into train, validation and test set.
sentences = read_text_file("zugoldseg.data")  
train_data, test_data = train_test_split(sentences, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

#Already a format for HMM, format =  [ [('Ukwengeza', 'V'), ('kulokhu', 'CDEM'), ...], ... ]
#Make a Format for CRF, format = [{"word": "Ukwengeza", "features": {...}, "label": "V"}, ...],
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        
    }
    if i > 0: 
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

#save a cleaned up data in the JSON file
with open("cleaned_data.json", "w") as json_file:      
    json.dump(sentences, json_file, indent=4)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAINING HMM>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Function to convert the data into sequences of observations and labels
def prepare_data_for_hmm(sentences):
    X = []
    y = []
    for sent in sentences:                                                              #traverse sentences
        obs = [token for token, _ in sent]                                              # Observations (tokens), make a list that takes only words and ignore tags
        labels = [label for _, label in sent]                                           # Labels (POS tags)
        X.append(obs)
        y.append(labels)
    return X, y                                                                         #x- ["ukugeza","ukuhamba"]     y-["N", "V"]

# Prepare training, validation, and test data
X_train, y_train = prepare_data_for_hmm(train_data)
X_val, y_val = prepare_data_for_hmm(val_data)
X_test, y_test = prepare_data_for_hmm(test_data)


# Create a mapping of labels (POS tags) to integers (for the HMM model)                 -> which label is which number
all_labels = list(set([label for sublist in y_train for label in sublist]))             #['N', 'VB', ...]
label_to_int = {label: i for i, label in enumerate(all_labels)}                         #['N':0 , 'VB' : 1]
int_to_label = {i: label for label, i in label_to_int.items()}                          #[0 :'N', 1 : 'VB']


# Convert labels to integers (HMM requires numeric labels)
y_train_int = [[label_to_int[label] for label in sent_labels] for sent_labels in y_train]       #['NN', 'VB', 'NN'] -> [0 , 1 , 0]
y_val_int = [[label_to_int[label] for label in sent_labels] for sent_labels in y_val]           #same thing to rest of dataset(train, val. test)
y_test_int = [[label_to_int[label] for label in sent_labels] for sent_labels in y_test]

# Convert observations (tokens) to integer indices (HMM requires numeric features)
# Here we treat each word as a separate observation 
word_count = Counter(word for sent in X_train for word in sent)                                     #count word frequencies

#keep words with atleast 5 occurrences
min_freq = 5
vocab = {word for word, count in word_count.items() if count >= min_freq}
word_to_int = {word: i for i, word in enumerate(vocab)}                                             #assign words to intergers(i), no repeation
word_to_int["<UNK>"] = len(word_to_int)                                                             # Add a special token for unknown words in the dictionary



X_train_int = [[word_to_int.get(word, word_to_int["<UNK>"]) for word in sent] for sent in X_train]  #assigning/selecting only the word integers now, like above, convert the whole list
X_val_int = [[word_to_int.get(word, word_to_int['<UNK>']) for word in sent] for sent in X_val]      #use <UNK> if finds an unknown word
X_test_int = [[word_to_int.get(word, word_to_int['<UNK>']) for word in sent] for sent in X_test]


# Initialize the HMM model
model = CategoricalHMM(n_components=len(all_labels), random_state=42, n_iter=100)

# Fit/train the model (train on the training data)
# HMM expects a list of sequences with the observations(tokens/words) and their corresponding states(POS)
X_train_int_flatten = np.concatenate([np.array(x).reshape(-1, 1) for x in X_train_int])                 #flatten x train int
lengths=[len(x) for x in X_train_int]                                                                   #get sequence length
model.fit(X_train_int_flatten, lengths)


#predict
X_val_int_flatten = np.concatenate([np.array(x).reshape(-1, 1) for x in X_val_int])
val_length = [len(x) for x in X_val_int]
y_val_pred_int = model.predict(X_val_int_flatten, val_length)


# Convert the predictions from integers back to labels.
start = 0
y_val_pred_HMM = []
for length in val_length:
    y_seg = y_val_pred_int[start:start+length]
    y_val_pred_HMM.append([int_to_label[i] for i in y_seg])
    start += length
        
    


# Print classification report for the validation set
# Flatten both lists
y_val_flat = [label for sent in y_val for label in sent]
y_val_pred_HMM_flat = [label for sent in y_val_pred_HMM for label in sent]

# Now call classification_report safely
print("Validation Set Classification Report:")
print(classification_report(y_val_flat, y_val_pred_HMM_flat))



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TESTINFG AND VALIDATION>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#same thing done above but now do it on the test values : predict, convert back, report.


# Test the model on the test set
Y_val_int_flatten = np.concatenate([np.array(x).reshape(-1, 1) for x in X_test_int])
test_lenght = [len(x) for x in X_test_int]

y_test_pred_int = model.predict(Y_val_int_flatten, test_lenght)

#Debug print to ensure alignment
print("Expected total sample :", Y_val_int_flatten.shape[0])
print("Sum of length array : ", sum(test_lenght))

assert sum(test_lenght) == Y_val_int_flatten.shape[0], "Mismatch between flattened array and lengths!"

#predict
y_test_pred_int = model.predict(Y_val_int_flatten, test_lenght)

# Convert the predictions from integers back to labels
start1 = 0
y_test_pred_HMM = []
for length in test_lenght:
    y_se = y_test_pred_int[start1:start1+length]
    y_test_pred_HMM.append([int_to_label[i] for i in y_se])
    start1 += length

# Filter out sentence pairs where prediction and ground truth lengths don't match
y_test_filtered = []
y_test_pred_filtered = []

for true_sent, pred_sent in zip(y_test, y_test_pred_HMM):
    if len(true_sent) == len(pred_sent):
        y_test_filtered.append(true_sent)
        y_test_pred_filtered.append(pred_sent)

# Flatten the filtered lists
y_test_flat = [label for sent in y_test_filtered for label in sent]
y_test_pred_HMM_flat = [label for sent in y_test_pred_filtered for label in sent]

        
# Print classification report for the test set
print("Test Set Classification Report:")
print(classification_report(y_test_flat, y_test_pred_HMM_flat, zero_division=1))


# Save the trained HMM model using pickle, to avoid having to train a model everytime you want to use it,
# and also to be able to use it in future.
with open("trained_hmm_model.pkl", "wb") as f:                 
    pickle.dump(model, f)



#-----------------------------------------------------------CRF---------------------------------------------------------------

# Extract features and labels for CRF
X_train_crf = [sent2features(sent) for sent in train_data]
y_train_crf = [sent2labels(sent) for sent in train_data]

X_val_crf = [sent2features(sent) for sent in val_data]
y_val_crf = [sent2labels(sent) for sent in val_data]

X_test_crf = [sent2features(sent) for sent in test_data]
y_test_crf = [sent2labels(sent) for sent in test_data]

# Initialize and train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',  # Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer
    c1=0.1,  # L1 penalty (regularization)
    c2=0.1,  # L2 penalty (regularization)
    max_iterations=100,  # Maximum iterations for convergence
    all_possible_transitions=True  # Allow all possible transitions between labels
)

# Train the CRF model
crf.fit(X_train_crf, y_train_crf)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>VALIDATION>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Evaluate the CRF model on the validation set
y_val_pred_CRF = crf.predict(X_val_crf)

# Calculate evaluation metrics for validation set
print("Validation Set Classification Report(CRF):")
print(metrics.flat_classification_report(y_val_crf, y_val_pred_CRF, zero_division=0))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TEST>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Evaluate the CRF model on the test set
y_test_pred_crf = crf.predict(X_test_crf)

# Calculate evaluation metrics for the test set
print("Test Set Classification Report(CRF):")
print(metrics.flat_classification_report(y_test_crf, y_test_pred_crf))


# Save the trained model if needed
with open("trained_crf_model.pkl", "wb") as f:                 
    pickle.dump(crf, f)


#--------------------------------------------HMM vs CRF----------------------------------------
def compare_models():
    # Flatten ground truth
    y_val_flat = [label for sent in y_val for label in sent]
    
    # Flatten HMM predictions
    y_val_pred_hmm_flat = [label for sent in y_val_pred_HMM for label in sent]

    # Flatten CRF predictions
    y_val_pred_crf_flat = [label for sent in y_val_pred_CRF for label in sent]

    # Accuracy
    acc_hmm = np.mean([yt == yp for yt, yp in zip(y_val_flat, y_val_pred_hmm_flat)])
    acc_crf = np.mean([yt == yp for yt, yp in zip(y_val_flat, y_val_pred_crf_flat)])

    print("\nModel Comparison Results:")
    if acc_hmm > acc_crf:
        print(f"HMM model performs better with accuracy: {acc_hmm}")
        print(f"Reason: HMM performs better in this case because...")
    else:
        print(f"CRF model performs better with accuracy: {acc_crf}")
        print(f"Reason: CRF captures more complex features due to its flexibility with transitions.")

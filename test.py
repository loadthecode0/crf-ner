from lc_crf import LinearChainCRF

c1 = LinearChainCRF()
sentence = ["Running", "in", "the", "morning", "."]
pos_tags = ["VBG", "IN", "DT", "NN", "."]
ner_labels = ["B-act", "O", "O", "B-tim", "O"]  # Ground-truth labels

import numpy as np

# Number of labels (NER classes)
N = 3  # Example: B-ACT (0), B-TIME (1), O (2)

# Number of POS-based features
P = 5  

# Number of additional state features (start_cap, end_ing, is_punct, etc.)
O = 6  

# Define weights manually (no randomness)
weights = np.array([
    # Transition Weights (Flattened N x N Matrix)
    -1.0,  0.5,  1.0,    # B-ACT → {B-ACT, B-TIME, O}
     0.3, -0.5,  0.7,    # B-TIME → {B-ACT, B-TIME, O}
     0.0,  1.2, -0.4,    # O → {B-ACT, B-TIME, O}

    # Initial Transition Weights (Start Probabilities)
     0.2, -0.1,  0.3,    # Start → {B-ACT, B-TIME, O}

    # Final Transition Weights (End Probabilities)
    -0.2,  0.4, -0.1,    # {B-ACT, B-TIME, O} → End

    # POS Emission Weights
     0.1,  0.2,  0.3,  0.4,  0.5,  # Weights for POS effects

    # Other State Feature Weights (start_cap, end_ing, is_punct, etc.)
     1.0,  2.0, -1.0,  0.5, -0.5, 0.2   # Weights for start_cap, end_ing, is_punct, etc.
])

# Print the structured weight vector
print("Manually Defined CRF Weights:")
print(weights)



X_train = [sentence]
pos_train = [pos_tags]
Y_train = [ner_labels]

c1.weights = weights
c1.num_ner = N
c1.num_pos = P
c1.all_NER_tags = list(set(ner_labels))
c1.all_POS = list(set(pos_tags))

c1.num_ner = len(c1.all_NER_tags)
c1.num_pos = len(c1.all_POS)

for idx, t in enumerate(c1.all_NER_tags):
    c1.ner_dict[t] = idx

for idx, t in enumerate(c1.all_POS):
    c1.pos_dict[t] = idx

print(c1.all_NER_tags)
print(c1.ner_dict)
print(c1.all_POS)
print(c1.pos_dict)

c1.score_seq(sentence, pos_tags, ner_labels)
print(c1.predict_viterbi(sentence, pos_tags))
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from collections import defaultdict
from joblib import Parallel, delayed
import pickle
import datetime
from utils import *
from torch.utils.data import DataLoader, TensorDataset


class LinearChainCRF(nn.Module):
    def __init__(self):
        super(LinearChainCRF, self).__init__()

        self.train_examples = {}
        self.all_NER_tags = set()
        self.all_POS = set()
        self.num_feats = 0

        self.pos_dict = defaultdict(int)
        self.ner_dict = defaultdict(int)
        self.num_ner = 0
        self.num_pos = 0

        # Observation functions
        self.obs_funcs = obs_funcs  # Import from utils.py 

        self.obs_feat_names = [
            'start_cap', 
            'end_ing', 
            'is_punct',
            'is_digit', 
            'is_start', 
            'is_end'
        ]

        # Placeholder for class weights
        self.class_weights = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def parse_train(self, filename: str, numlines=None):
        parsed_data, all_NER_tags, all_POS = parse(filename, numlines)
        self.all_NER_tags = all_NER_tags
        self.all_POS = all_POS
        self.num_ner = len(self.all_NER_tags)
        self.num_pos = len(self.all_POS)

        for idx, t in enumerate(self.all_NER_tags):
            self.ner_dict[t] = idx
        for idx, t in enumerate(self.all_POS):
            self.pos_dict[t] = idx

        self.train_examples = parsed_data

    def init_weights(self):
        """Initialize learnable weights as a PyTorch tensor."""
        num_transitions = self.num_ner ** 2 + 2 * self.num_ner
        num_emissions = self.num_ner * self.num_pos
        num_observations = self.num_ner * len(self.obs_funcs)
        
        self.num_feats = num_transitions + num_emissions + num_observations
        self.weights = nn.Parameter(torch.randn(self.num_feats, device=self.device) * 0.01)  # Move to GPU

    def emission_score(self, Y:List[str]=None, X:List[str]=None, pos_seq:List[str]=None, t:int=None, T:int=None, y:str=None, y_:str=None):
        """Computes emission score using torch operations."""
        em = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        n = self.num_ner
        n2 = n ** 2
        p = self.num_pos

        T = len(X)

        if t < T:
            
            if y is None:
                y = Y[t]
            i = self.pos_dict[pos_seq[t]]
            j = self.ner_dict[y]
            o = len(self.obs_funcs)
            em += self.weights[n2 + 2*n + i*n + j]  # POS emission

            em_wts = self.weights[n2 + 2*n + n*p + j*o : n2 + 2*n + n*p + j*o + o]
            args = (X[t], pos_seq[t], t, len(X), y, None)
            # em_feats = torch.zeros((n, len(self.obs_funcs)), dtype=torch.float32, requires_grad=False)
            em_feats = torch.tensor([f(*args) for f in self.obs_funcs], dtype=torch.float32, requires_grad=False)

            em += torch.dot(em_wts, em_feats)  # Other observation functions
        # return em * self.class_weights.get(y, 1.0)
        return em * 0.5

    def transition_score(self, Y:List[str]=None, t:int=None, T:int=None, y:str=None, 
                         y_:str=None, O_penalty=0.5, entity_boost=1.5, reg_weight=4.0):
        """Computes transition score using PyTorch tensors."""
        n = self.num_ner
        n2 = n ** 2

        # only if y and y_ are not specified
        if y == None and t<T:
            y = Y[t]
        if y_ == None and t>0: 
            y_ = Y[t-1]

        if t == 0:
            base = self.weights[self.ner_dict[y] + n2] # BOS -> y
        elif t == T:
            base = self.weights[self.ner_dict[y_] + n2 + n] # y_ -> EOS
        else:
            base = self.weights[self.ner_dict[y] + n * self.ner_dict[y_]] # General transition

        if t < T:
            scale = self.class_weights.get(y, 1.0) * self.class_weights.get(y_, 0.01)
            return base * scale * 2.0
        elif t==T :
            scale = self.class_weights.get(y_, 1.0) * 0.01
            return base * scale * 2.0
        
        return base 
    
    def score_seq(self, X: List[str], pos_seq: List[str], Y: List[str]) -> torch.Tensor:
        """
        Computes the score of a sequence Y given the observations X.

        Inputs:
            X : List[str] -> Token sequence
            pos_seq : List[str] -> POS tags sequence
            Y : List[str] -> NER labels sequence
        
        Returns:
            score_X_Y : torch.Tensor -> Log-score of the sequence
        """

        T = len(X)  # number of tokens in sequence
        score_X_Y = torch.tensor(0.0, dtype=torch.float32) 

        for t in range(0, T + 1):  # iterate from t=0 to T (including EOS transition)
            score_X_Y += (
                self.transition_score(t=t, T=T, y=(Y[t] if t < T else None), y_= (Y[t-1] if t > 0 else None)) +
                self.emission_score(X=X, pos_seq=pos_seq, t=t, y=(Y[t] if t < T else None))
            )

        return score_X_Y


    def forward_partition(self, X:List[str],pos_seq:List[str] ):
        """Computes log partition function using forward algorithm."""
        
        T = len(X)
        n = self.num_ner
        dp = torch.full((T + 1, n + 1), -float('inf'), device=self.device)  # Move DP table to GPU

        # try: # when y, y_ are not given explicity
        #     if y == None and t<T : 
        #         y = Y[t]
        #     if y_ == None and t>0 : 
        #         y_ = Y[t-1]
        # except:
        #     print(T, t, X, Y, len(X), len(pos_seq), len(Y))

        for y in self.all_NER_tags:
            j = self.ner_dict[y]
            dp[0][j] = self.transition_score(t=0, T=T, y=y, y_=None) + self.emission_score(X=X, pos_seq=pos_seq, t=0, T=T, y=y, y_=None)

        for t in range(1, T):
            for y in self.all_NER_tags:
                j = self.ner_dict[y]
                dp[t][j] = torch.logsumexp(
                    torch.tensor([
                        dp[t-1][self.ner_dict[y_]] +
                        self.transition_score(t=t, T=T, y=y, y_=y_) + \
                        self.emission_score(X=X, pos_seq=pos_seq, t=t, T=T, y=y, y_=None) 
                        for y_ in self.all_NER_tags
                    ]),
                    dim=0
                )

        log_Z = torch.logsumexp(
            torch.tensor([
                dp[T-1][self.ner_dict[y_]] + self.transition_score(t=T, T=T, y=None, y_=y_)
                for y_ in self.all_NER_tags
            ]),
            dim=0
        )

        return log_Z, dp

    def nll(self, X_train: List[List[str]], pos_train: List[List[str]], Y_train: List[List[str]], 
            reg_lambda=0.0, O_penalty=0.75, entity_boost=1.5):
        """Computes the Negative Log-Likelihood (NLL) loss function."""
        
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for X, pos_seq, Y in tqdm(zip(X_train, pos_train, Y_train)):
            score = self.score_seq(X, pos_seq, Y).to(self.device)
            log_Z = (self.forward_partition(X, pos_seq)[0]).to(self.device)
            loss += log_Z - score

        loss /= len(X_train)  
        # loss += reg_lambda * torch.norm(self.weights, p=2)  # L2 regularization
        return loss 

    def train(self, use_class_wts=True, max_iter=100, train=True):
        """Trains the CRF model using L-BFGS optimizer."""
        X_train = [[word for word in sentence] for sentence in self.train_examples['Tokens'].tolist()]
        pos_train = [[tag for tag in pos_seq] for pos_seq in self.train_examples['POS'].tolist()]
        Y_train = [[tag for tag in labels] for labels in self.train_examples['NER_tags'].tolist()]

        # # Move to GPU
        # X_train = [[torch.tensor(word, device=self.device) for word in sentence] for sentence in X_train]
        # pos_train = [[torch.tensor(tag, device=self.device) for tag in pos_seq] for pos_seq in pos_train]
        # Y_train = [[torch.tensor(tag, device=self.device) for tag in labels] for labels in Y_train]



        if use_class_wts:
            self.class_weights = compute_class_weights(Y_train)
        optimizer = optim.LBFGS([self.weights], lr=0.1, max_iter=3)

        def closure():
            optimizer.zero_grad()
            loss = self.nll(X_train, pos_train, Y_train)
            loss.backward()
            return loss

        if not train: 
            print(f'not training')
            plot_weights(self.weights, self.num_ner, self.num_pos, len(self.obs_funcs), self.all_NER_tags, self.all_POS, self.obs_feat_names)
            return
        
        print(f'Training begins')

        for i in tqdm(range(max_iter)):
            optimizer.step(closure)
            print(f"Iteration {i+1}: Loss = {closure().item():.4f}")
            # save_crf_model('test', self, f'checkpoint_iter_{i+1}_loss_{100*(closure().item()):4.0f}')

        plot_weights(self.weights, self.num_ner, self.num_pos, len(self.obs_funcs), self.all_NER_tags, self.all_POS, self.obs_feat_names)

    def fit(self, filename:str, numlines:int=None, use_class_wts:int=None, show_tqdm:bool=False, max_iter:int=5, train:bool=True) -> None:
        self.parse_train(filename=filename, numlines=numlines)
        self.init_weights()
        self.use_tqdm = show_tqdm
        self.train(use_class_wts, max_iter, train)

    def predict_viterbi(self, obs: List[str], pos_seq: List[str]):
        """Decodes sequence using Viterbi algorithm."""
        T = len(obs)
        n = self.num_ner
        dp = torch.full((T + 1, n + 1), -float('inf'), device=self.device)
        trace = torch.zeros((T + 1, n + 1), dtype=torch.long, device=self.device)


        for y in self.all_NER_tags:
            j = self.ner_dict[y]
            dp[0][j] = (
                self.transition_score(y=y, t=0, T=T) + 
                self.emission_score(X=obs, pos_seq=pos_seq, t=0, T=T, y=y, y_=None)
            )

        for t in range(1, T):
            for y in self.all_NER_tags:
                j = self.ner_dict[y]
                best = -float('inf')
                back = 0
                for y_ in self.all_NER_tags:
                    j_ = self.ner_dict[y_]
                    new_score = (dp[t-1][j_] + 
                                 self.transition_score(t=t, T=T, y=y, y_=y_) + 
                                 self.emission_score(X=obs, pos_seq=pos_seq, t=t, T=T, y=y, y_=None) 
                    )
                    if new_score > best:
                        best = new_score
                        back = j_
                dp[t][j] = best
                trace[t][j] = back

        for y_ in self.all_NER_tags:
            j_ = self.ner_dict[y_]

            new_score = dp[T-1][j_] + self.transition_score(t=T, T=T, y_=y_) 
            if new_score > best:
                best = new_score
                back = j_

        dp[T][n] = best
        trace[T][n] = back

        pred_labels = []
        t = T
        j = n

        while t>0:
            try:
                j = int(trace[t][j])
                pred_labels.append(self.all_NER_tags[j])
                t -= 1
            except:
                print(f"Error at index {t, j}")
                print(f"--> {self.all_NER_tags[j]}")

        return pred_labels[::-1] # reversed

    def eval(self, Y_pred: List[List[str]], Y_test: List[List[str]]):
        """
        Evaluates model predictions using Precision, Recall, F1-score, and Accuracy.

        Inputs:
            Y_pred : List[List[str]] -> Predicted labels for each sequence.
            Y_test : List[List[str]] -> True labels for each sequence.

        Returns:
            accuracy : float -> Overall accuracy
            precision : dict -> Per-class precision
            recall : dict -> Per-class recall
            f1_score : dict -> Per-class F1-score
        """
        assert len(Y_test) == len(Y_pred), "Mismatch in number of sequences"

        # Flatten lists into single tensor sequences
        Y_test_flat = sum(Y_test, [])  # Equivalent to list flattening
        Y_pred_flat = sum(Y_pred, [])

        assert len(Y_test_flat) == len(Y_pred_flat), "Mismatch in number of labels after flattening"

        unique_labels = set(Y_test_flat) | set(Y_pred_flat)
        label_counts = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in unique_labels}

        # Compute confusion counts
        for true, pred in zip(Y_test_flat, Y_pred_flat):
            if true == pred:
                label_counts[true]['TP'] += 1
            else:
                label_counts[pred]['FP'] += 1
                label_counts[true]['FN'] += 1

        # Compute precision, recall, and F1-score for each label
        precision, recall, f1_score = {}, {}, {}
        for label, counts in label_counts.items():
            tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
            precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0.0

        # Compute overall metrics
        all_tp = sum(counts['TP'] for counts in label_counts.values())
        all_fp = sum(counts['FP'] for counts in label_counts.values())
        all_fn = sum(counts['FN'] for counts in label_counts.values())

        accuracy = all_tp / len(Y_test_flat)
        precision_all = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall_all = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1_score_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if (precision_all + recall_all) > 0 else 0.0

        # Print summary
        print(f"Accuracy: {accuracy:.10f}")
        print("\nLabel-wise Metrics:")
        for label in unique_labels:
            print(f"Label: {label}, Precision: {precision[label]:.10f}, Recall: {recall[label]:.10f}, F1-Score: {f1_score[label]:.10f}")

        return accuracy, precision, recall, f1_score
    
    def eval_from_file(self, filename: str, numlines: int = None):
        """
        Evaluates the model on a dataset from a file.

        Inputs:
            filename : str -> Path to the dataset file.
            numlines : int (optional) -> Number of lines to read from the file.
        """
        parsed_data, _, _ = parse(filename=filename, numlines=numlines)

        X_test = parsed_data['Tokens'].tolist()
        pos_test = parsed_data['POS'].tolist()
        Y_test = parsed_data['NER_tags'].tolist()

        # print(X_test)
        N = len(X_test)

        Y_pred = []
        for i in tqdm(range(N)):
            y_pred = self.predict_viterbi(X_test[i], pos_test[i])
            Y_pred.append(y_pred)

            # print(f"Predicted: {y_pred}\nActual: {Y_test[i]}\n")

        df = pd.DataFrame({'Tokens': X_test, 'POS': pos_test, 'Y_test': Y_test, 'Y_pred' : Y_pred})
        df.to_csv('output.csv', index=False)
        # evaluate predictions
        self.eval(Y_pred=Y_pred, Y_test=Y_test)

def save_crf_model(dir, crf_model, extra: str):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{dir}/crf_{timestamp}_{extra}.pt"

    torch.save(crf_model.state_dict(), filename, _use_new_zipfile_serialization=False)

    print(f"CRF model saved to {filename}")

def load_crf_model(model_class, filename: str):
    model = model_class()
    k = torch.load(filename, weights_only=True)
    # print(k.keys())
    model.load_state_dict(k, strict=False)
    print(f"CRF model loaded from {filename}")
    return model

def test():
    c = LinearChainCRF()
    c.fit('data/ner_train.csv', use_class_wts= True, numlines=1000, show_tqdm=False, max_iter=5) # modify numlines for the appropriate training set size and epochs
    c.eval_from_file('data/ner_test.csv', numlines=1000) # modify numlines for the appropriate testing set size


                     
if __name__ == "__main__":
    test()
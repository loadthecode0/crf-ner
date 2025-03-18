from utils import *
from typing import DefaultDict, Set, List
from collections import defaultdict
from joblib import Parallel, delayed
class LinearChainCRF:

    def __init__(self):
        self.train_examples = {}
        self.all_NER_tags = set()
        self.all_POS = set()
        self.num_feats = 0

        self.pos_dict = defaultdict(int)
        self.ner_dict = defaultdict(int)
        self.num_ner = 0
        self.num_pos = 0


        # self.id_funcs = []

        # self.obs_funcs = [
        #     start_cap,
        #     end_ing
        # ]

        self.obs_funcs = obs_funcs #imported from utils.py 

        self.obs_feat_names = [
            'start_cap',
            'end_ing',
            'is_punct',
            'is_digit',
            'is_start',
            'is_end'
        ]

        # self.feat_funcs = []

    def parse_train(self, filename:str, numlines=None) -> None:
        
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

    # def build_id_funcs(self) -> None:
    #     for ner_tag in self.all_NER_tags:
    #         f = lambda t : t == ner_tag
    #         self.id_funcs.append(f)

    # def build_feature_funcs(self) -> None:
    #     for i in self.id_funcs:
    #         for q in self.obs_funcs:
    #             self.feat_funcs.append(lambda y,x : i(y)*q(x))

    # def build_dataframe(self) -> None:
    #     cols = ['Sentence ID', 'Token', 'POS']
    #     for t in self.all_NER_tags:
    #         for q in self.obs_feat_names:
    #             cols.append(f'Id_{t}*q_{q}')

    #     cols.append('Label')
    #     self.data = pd.DataFrame(columns=cols)

    #     print(self.train_examples)
    #     print(len(self.train_examples))
    #     i = 0
    #     for id in self.train_examples.keys():
    #         print(i)
    #         i+=1
    #         egs = self.train_examples[id]
    #         l = [id]*len(egs['Tokens'])
    #         egs['Sentence ID'] = l
    #         self.data = pd.concat([self.data, pd.DataFrame(egs)], ignore_index=True)

    #     print(self.data.head(n=25))

    def emission_score(self, Y:List[str]=None, X:List[str]=None, pos_seq:List[str]=None, t:int=None, T:int=None, y:str=None, y_:str=None) -> np.float16:
        
        """
        calculates the emission log prob (which includes POS given NER tag, and binary obs functions)
        if t==T, it means the sentence ended, and there is 0 contribution to score in log space (1 prob of EOS emission)
        """

        em = 0.0
        n = self.num_ner
        n2 = n**2
        p = self.num_pos

        if t < T:

            em_wts = self.weights[n2 + 2*n + n*p :]

            if y is None:
                y = Y[t]
            args = (X[t], pos_seq[t], t, T, y, y_)
            em_feats = [float(f(*args)) for f in self.obs_funcs]
            # score from only POS emission
            i = self.pos_dict[pos_seq[t]]
            j = self.ner_dict[y]
            em += self.weights[n2 + 2*n + i*n + j] # for current pos -> conditioned on y
            # score from other obs functions
            em += np.dot(em_wts, em_feats)

        return em
    
    def transition_score(self, Y:List[str]=None, t:int=None, T:int=None, y:str=None, 
                         y_:str=None, O_penalty=0.5, entity_boost=1.5, reg_weight=4.0) -> np.float16:
        idx = 0
        n = self.num_ner
        n2 = n**2
        p = self.num_pos

        # only if y and y_ are not specified
        if y == None and t<T:
            y = Y[t]
        if y_ == None and t>0: 
            y_ = Y[t-1]

        if t==0:
            base = self.weights[self.ner_dict[y] + n2]# BOS -> y
        elif t==T:
            base =  self.weights[self.ner_dict[y_] + n2 + n]# y_ -> EOS, y is EOS
        else: 
            base = self.weights[self.ner_dict[y] + n*self.ner_dict[y_]]#for general transition score

        # # Apply O->O penalty using regularization scaling
        # if y_ == "O" and y == "O":
        #     base += reg_weight * abs(np.log(O_penalty))  # Strong penalization of O->O

        # # Boost O->B-* transitions
        # elif y_ == "O" and y is not None and y.startswith("B-"):
        #     base -= reg_weight * np.log(entity_boost)  # Encourage O->Entity transitions
        if t<T:
            base *= self.class_weights[y]

        return base

    def make_features(self, X:List[str], pos_seq:List[str], t:int, T:int, Y:List[str]=None, y:str=None, y_:str=None):
        """
        Inputs:
            y : str       -> current label, y_t (hidden state)
           y_ : str       -> prev label, y_(t-1) (hidden state)
            X : List[str] -> obs sequence (tokens list)
      pos_seq : List[str] -> obs sequence (POS list)
            t : int       -> timestep
            T : int       -> total timesteps (length of sentence)
        """

        try: # when y, y_ are not given explicity
            if y == None and t<T : 
                y = Y[t]
            if y_ == None and t>0 : 
                y_ = Y[t-1]
        except:
            print(T, t, X, Y, len(X), len(pos_seq), len(Y))

        # print(f"Hi {self.num_feats}")
        feats = np.zeros(self.num_feats)
        n = self.num_ner
        n2 = n**2
        p = self.num_pos
        n_p = n*p

        if t==0: 
            feats[self.ner_dict[y] + n2] = 1.0 # BOS -> y
        elif t==T:
            feats[self.ner_dict[y_] + n2 + n] = 1.0 # y_ -> EOS, y is EOS
        else:
            feats[self.ner_dict[y] + n*self.ner_dict[y_]] = 1.0 #for transition score
        
        if t < T: #pos tag only for valid timesteps, otherwise index error
            args = (X[t], pos_seq[t], t, T, y, y_)
            em_feats = [f(*args) for f in self.obs_funcs]
            # score from only POS emission
            i = self.pos_dict[pos_seq[t]]
            j = self.ner_dict[y]
            feats[n2 + 2*n + i*n + j] = 1.0# for current pos -> conditioned on y
            # score from other obs functions
            feats[n2 + 2*n + n*p :] = em_feats

        return feats

    def score_seq(self, X:List[str], pos_seq:List[str], Y:List[str]) -> np.float16:
        """
        Inputs:
            X : List[str] -> obs sequence (tokens list)
            pos_seq : List[str] -> obs sequence (POS list)
            Y : List[str] -> hidden sequence (NER tags list)
        """
        T = len(X)
        score_X_Y = 0.0

        for t in range(0, T+1):
            # score_X_Y += np.dot(self.weights, self.make_features(Y, X, pos_seq, t, T))
            score_X_Y += \
                self.transition_score(Y, t, T) + \
                self.emission_score(Y, X, pos_seq, t, T) 
                
            # if t<T:
            #     score_X_Y *= self.class_weights[Y[t]]
            

            # print(f"t=\t{t} ---> + {self.transition_score(Y, t, T)} + {self.emission_score(Y, X, pos_seq, t, T)}")

        # print(score_X_Y)
        return score_X_Y

    def forward_partition(self, X:List[str],pos_seq:List[str] ) -> np.float16:
        T = len(X)
        n = self.num_ner
        n2 = n**2
        p = self.num_pos
        o = len(self.obs_funcs)

        dp = np.zeros((T+1, n+1)) 

        for y in self.all_NER_tags:
            j = self.ner_dict[y] # get index of curr NER label
            # only BOS -> y transitions, no need to consider logaddexp
            dp[0][j] = self.transition_score(y=y, t=0, T=T) + self.emission_score(X=X, pos_seq=pos_seq, t=0, T=T, y=y, y_=None)

        for t in range(1, T): # goes till T-1, ie before EOS
            for y in self.all_NER_tags:
                j = self.ner_dict[y]

                dp[t][j] = \
                np.logaddexp.reduce([
                    dp[t-1][self.ner_dict[y_]] + \
                    self.transition_score(t=t, T=T, y=y, y_=y_) + \
                    self.emission_score(X=X, pos_seq=pos_seq, t=t, T=T, y=y, y_=None) 
                    for y_ in self.all_NER_tags
                ])

        #nth label = EOS (y_ -> EOS)
        dp[T][n] = np.logaddexp.reduce([
            dp[T-1][self.ner_dict[y_]] + \
            self.transition_score(t=T, T=T, y_=y_) 
            for y_ in self.all_NER_tags ])

        return dp[T][n], dp
    

    def backward_partition(self, X: List[str], pos_seq: List[str]) -> np.float16:
        """
        Computes the Backward Algorithm for CRF.
        Returns log Z(X) using dynamic programming.
        """

        T = len(X)  # Sequence length
        n = self.num_ner  # Number of NER labels
        n2 = n**2  # Total transition weights
        p = self.num_pos  # Number of POS tags
        o = len(self.obs_funcs)  # Number of observation features

        # Initialize DP table
        dp = np.zeros((T+1, n+1))  # (T+1) x (n+1) matrix

        # Base case: EOS (end of sentence)
        for y in self.all_NER_tags:
            j = self.ner_dict[y]
            dp[T][j] = self.transition_score(t=T, T=T, y_=y)  # Transition to EOS

        # Recursively compute backward probabilities
        for t in reversed(range(T)):  # Iterate from T-1 down to 0
            for y in self.all_NER_tags:
                j = self.ner_dict[y]

                dp[t][j] = np.logaddexp.reduce([
                    dp[t+1][self.ner_dict[y_next]] +  # Next step probability
                    self.transition_score(t=t+1, T=T, y=y_next, y_=y)[0] +  # Transition y → y_next
                    self.emission_score(X=X, pos_seq=pos_seq, t=t+1, T=T, y=y_next, y_=None)  # Emission score
                    for y_next in self.all_NER_tags
                ])

        # Sum over all BOS → y transitions to compute log Z(X)
        log_Z = np.logaddexp.reduce([
            dp[0][self.ner_dict[y]] + self.transition_score(y=y, t=0, T=T) + 
            self.emission_score(X=X, pos_seq=pos_seq, t=0, T=T, y=y, y_=None)
            for y in self.all_NER_tags
        ])

        return dp

    
    # def nll(self, W, X_train: List[List[str]], pos_train: List[List[str]], Y_train: List[List[str]], reg_lambda=0.1) -> float:
    #     """Negative Log-Likelihood with Parallel Processing"""
    #     self.weights = W
    #     N = len(X_train)

    #     def compute_log_likelihood(i):
    #         X, pos_seq, Y = X_train[i], pos_train[i], Y_train[i]
    #         return self.score_seq(X, pos_seq, Y) - self.forward_partition(X, pos_seq)

    #     r = range(0,N)
    #     if self.use_tqdm:
    #         r = tqdm(r)
    #     # parallel computation
    #     ll_values = Parallel(n_jobs=-1)(delayed(compute_log_likelihood)(i) for i in r)

    #     # mean
    #     n = self.num_ner
    #     num_trans_wts = n**2 + 2*n
    #     return (-sum(ll_values) / N) + reg_lambda * np.sum(self.weights[:num_trans_wts]**2)


    def nll(self, W, X_train: List[List[str]], pos_train: List[List[str]], Y_train: List[List[str]], 
            reg_lambda=0.5, O_penalty=0.75, entity_boost=1.5) -> float:
        """Negative Log-Likelihood with Weighted Loss (Prevents 'O' Overprediction)"""
        self.weights = W
        N = len(X_train)

        def compute_log_likelihood(i):
            """Computes sequence score while applying weighted penalties for 'O' and entity labels."""
            X, pos_seq, Y = X_train[i], pos_train[i], Y_train[i]

            # Compute sequence score
            seq_score = self.score_seq(X, pos_seq, Y)
            log_partition = self.forward_partition(X, pos_seq)[0]

            

            # # Adjust 'O' and entity label influence
            # for label in Y:
            #     if label == "O":
            #         seq_score *= O_penalty  # Reduce 'O' impact
            #     else:
            #         seq_score *= entity_boost  # Boost entity learning

            return seq_score - log_partition

        # Use tqdm for progress bar
        r = range(0, N)
        if self.use_tqdm:
            r = tqdm(r)

        # Parallel computation
        ll_values = Parallel(n_jobs=-1)(delayed(compute_log_likelihood)(i) for i in r)

        # Compute negative log-likelihood + L2 regularization
        n = self.num_ner
        num_trans_wts = n**2 + 2*n
        return (-sum(ll_values) / N) + reg_lambda * np.sum(self.weights**2)
    
    def expected_feature_counts(self, X, pos_seq):
        """Computes expected feature counts using model probabilities."""
        feature_vector = np.zeros_like(self.weights)
        T = len(X)  # Sequence length
        n = self.num_ner
        n2 = n**2
        p = self.num_pos
        n_p = n*p

        # Compute forward and backward probabilities
        log_Z, alpha = self.forward_partition(X, pos_seq)  # Forward probabilities
        beta = self.backward_partition(X, pos_seq)  # Backward probabilities

        T = len(X)
        n = self.num_ner
        n2 = n**2
        p = self.num_pos
        o = len(self.obs_funcs)
        n_p = n*p

        dp = np.zeros((T+1, n+1)) 

        for y in self.all_NER_tags:
            j = self.ner_dict[y] # get index of curr NER label
            # only BOS -> y transitions, no need to consider logaddexp
            feature_vector[j + n2] = (
                self.transition_score(y=y, t=0, T=T) + 
                self.emission_score(X=X, pos_seq=pos_seq, t=0, T=T, y=y, y_=None) +
                beta[t, j] - 
                log_Z
            )

        for t in range(1, T): # goes till T-1, ie before EOS
            for y in self.all_NER_tags:
                j = self.ner_dict[y]
                for y_ in self.all_NER_tags:
                    j_ = self.ner_dict[y_]

                feature_vector[j + n*j_] = \
                np.exp(
                    alpha[t-1][j_] + 
                    self.transition_score(t=t, T=T, y=y, y_=y_) + 
                    self.emission_score(X=X, pos_seq=pos_seq, t=t, T=T, y=y, y_=None) +
                    beta[t][j] - 
                    log_Z
                )

        #nth label = EOS (y_ -> EOS)
        dp[T][n] = np.logaddexp.reduce([
            dp[T-1][self.ner_dict[y_]] + \
            self.transition_score(t=T, T=T, y_=y_) 
            for y_ in self.all_NER_tags ])




        for t in range(T):
            for y in range(self.num_labels):
                # Compute state feature expectations
                args = (X[t], pos_seq[t], t, T, y, None)
                state_features = [f(*args) for f in self.obs_funcs]
                marginal_prob = np.exp(alpha[t][y] + beta[t][y] - log_Z)  # P(y_t | X)
                feature_vector[n2 + 2*n + n_p :] += marginal_prob * state_features  # Weighted sum
                # score from only POS emission
                i = self.pos_dict[pos_seq[t]]
                j = self.ner_dict[y]
                feature_vector[n2 + 2*n + i*n + j] = marginal_prob# for current pos -> conditioned on y

                # Compute transition feature expectations (except first word)
                if t > 0:
                    for y_prev in range(self.num_labels):
                        trans_idx = y_prev * self.num_labels + y
                        trans_prob = np.exp(alpha[t-1, y_prev] + self.weights[len(state_features) + trans_idx] + beta[t, y] - Z)
                        feature_vector[len(state_features) + trans_idx] += trans_prob  # Update transition counts

        return feature_vector


    def compute_gradient(self, W, X_train, pos_train, Y_train, reg_lambda=0.5):
        """Computes the gradient of the Negative Log-Likelihood (Jacobian)"""
        self.weights = W  # Update weights
        N = len(X_train)
        gradient = np.zeros_like(W)

        def compute_per_sequence_gradient(i):
            """Computes the gradient for a single training sequence"""
            X, pos_seq, Y = X_train[i], pos_train[i], Y_train[i]
            T = len(X)

            # Compute feature expectations
            observed_features = np.sum([self.make_features(Y, X, pos_seq, t, T) for t in range(T)])
            expected_features = self.expected_feature_counts(X, pos_seq)

            # Compute per-sequence gradient
            return observed_features - expected_features

        # Parallel computation
        gradients = Parallel(n_jobs=-1)(delayed(compute_per_sequence_gradient)(i) for i in range(N))

        # Aggregate gradients over all sequences
        for g in gradients:
            gradient += g

        # Apply L2 Regularization
        gradient += 2 * reg_lambda * self.weights

        return -gradient / N  # Negative gradient for minimization

    def callback_function(self, weights):
        """Callback function to print loss during training."""
        loss = self.nll(weights, self.X_train, self.pos_train, self.Y_train)
        print(f"Current NLL Loss: {loss:.4f}, ||W|| = {np.sqrt(np.sum(self.weights**2))}")

    def train(self, batchsize:int=50, maxiter:int=10, use_dummy_wts:bool=False, dummy_wts=None, train:bool=True) -> None:

        # initialize weights

        self.num_feats = (self.num_ner)**2 + 2*(self.num_ner) + self.num_ner*self.num_pos + len(self.obs_funcs)
        # first set of weights is transition score/prob - [0, n*n-1]
        # second set of weights is transition from BOS to NER tags, and NER tags to EOS - [n*n, n*n + 2n - 1]
        # third set of weights is for current POS tag being emitted from NER label y - [n*n + 2n, n*n + 2n + p - 1]
        # fourth set of wights is for miscellaneous obs funcs - [n*n + 2n + p, n*n + 2n + p + o - 1]
        mu = 0.0
        sigma = 0.01

        if not use_dummy_wts:
            self.weights = sigma*np.random.randn(self.num_feats) + np.array([mu]*self.num_feats)
        else:
            self.weights = dummy_wts
        print(self.num_feats)
        print(self.weights)


        if not train: 
            print(f'not training')
            plot_weights(self.weights, self.num_ner, self.num_pos, len(self.obs_funcs), self.all_NER_tags, self.all_POS, self.obs_feat_names)
            return

        # plt.hist(self.weights, bins=10, edgecolor='black', alpha=0.7)
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Data')
        # plt.show()

        # send training examples to scoring function
        # self.train_examples = self.train_examples.reset_index() 
        # print(self.train_examples)
        # for index, eg in self.train_examples.iterrows():
        #     print(eg)
        #     X = eg['Tokens']
        #     Y = eg['NER_tags']
        #     pos_seq = eg['POS']
        #     self.score_seq(X, pos_seq, Y)

        X_train = self.train_examples['Tokens'].tolist()
        pos_train = self.train_examples['POS'].tolist()
        Y_train = self.train_examples['NER_tags'].tolist()

        # print(X_train)
        # print(pos_train)
        # print(Y_train)
        self.class_weights = compute_class_weights(Y_train)
        print(self.class_weights)

        """ Train using L-BFGS optimizer """
        self.X_train = X_train
        self.Y_train = Y_train
        self.pos_train = pos_train

        N = len(X_train)
        batch_size = batchsize # train on 10 examples at a time
        for i in range(0, len(X_train), batch_size):
            m = min(i+batch_size, N)
            batch_X, batch_pos, batch_Y = X_train[i:m], pos_train[i:m], Y_train[i:m]
            result = minimize(
                self.nll, 
                self.weights, 
                args=(batch_X, batch_pos, batch_Y), 
                method='L-BFGS-B',
                callback=self.callback_function,
                options={'maxiter': maxiter, 'disp':True}
            )
            self.weights = result.x  # update weights

            # if (i%50 == 0):
            #     self.callback_function(self.weights)

        # result = minimize(self.nll, self.weights, args=(X_train, pos_train, Y_train),
        #                   method='L-BFGS-B', callback=self.callback_function, options={'disp': True, 'maxiter':maxiter})
        # self.weights = result.x  # Update model parameters
        print(self.weights)
        plot_weights(self.weights, self.num_ner, self.num_pos, len(self.obs_funcs), self.all_NER_tags, self.all_POS, self.obs_feat_names)


    def fit(self, filename:str, numlines:int=None, batchsize:int=None, show_tqdm:bool=False, maxiter:int=10, train:bool=True) -> None:
        self.parse_train(filename=filename, numlines=numlines)
        
        # self.build_id_funcs()
        # self.build_feature_funcs()
        # self.build_dataframe()
        self.use_tqdm = show_tqdm
        self.train(batchsize, maxiter, train=train)

    

    def predict_viterbi(self, obs:List[str], pos_seq:List[str]) -> List[str]:
        T = len(obs)
        n = self.num_ner
        n2 = n**2
        p = self.num_pos
        o = len(self.obs_funcs)

        dp = np.full((T+1, n+1), -np.inf)
        trace = np.zeros((T+1, n+1))

        for y in self.all_NER_tags:
            j = self.ner_dict[y]
            # print(f't=0')
            # print(f'Checking y={j} and y_=BOS')
            dp[0][j] = self.transition_score(y=y, t=0, T=T) + self.emission_score(X=obs, pos_seq=pos_seq, t=0, T=T, y=y, y_=None)


        for t in range(1, T): # goes till T-1, ie before EOS
            for y in self.all_NER_tags:
                j = self.ner_dict[y]

                best = -np.inf
                back = 0

                # print(f't={t}')
                for y_ in self.all_NER_tags:
                    j_ = self.ner_dict[y_]

                    # new_score = dp[t-1][j_] + self.emission_score([], obs, pos_seq, t, T, y=y) + self.transition_score([],obs,pos_seq, t, T, y=y, y_=y_)
                    # print(f'> Checking y={j} and y_={y_}')
                    new_score = dp[t-1][j_] + \
                        self.transition_score(t=t, T=T, y=y, y_=y_) + \
                        self.emission_score(X=obs, pos_seq=pos_seq, t=t, T=T, y=y, y_=None) 
                    # print(f'> new_score = {new_score}')
                    if new_score > best:
                        best = new_score
                        back = j_

                dp[t][j] = best
                trace[t][j] = back

                # print(f'Viterbi for t={t}, y={y}, j={j}, {dp[t][j]}, trace={trace[t][j]} -> {self.all_NER_tags[int(trace[t][j])]}')

         #nth = EOS
        # print(f't={t}')
        for y_ in self.all_NER_tags:
            j_ = self.ner_dict[y_]

            new_score = dp[T-1][j_] + self.transition_score(t=T, T=T, y_=y_) 
            if new_score > best:
                best = new_score
                back = j_

        dp[T][n] = best
        trace[T][n] = back
        # print(f'Viterbi for t={T}, y=EOS, j={n}, {dp[T][n]}, trace={trace[T][n]} -> {self.all_NER_tags[int(trace[T][n])]}')

        pred_labels = []
        t = T
        j = n
        # j = int(trace[T][n])

        while t>0:
            try:
                j = int(trace[t][j])
                pred_labels.append(self.all_NER_tags[j])
                t -= 1
            except:
                print(f"Error at index {t, j}")
                print(f"--> {self.all_NER_tags[j]}")

        return pred_labels[::-1] # reversed

    def eval(self, Y_pred:List[List[str]], Y_test:List[List[str]]):
        assert len(Y_test) == len(Y_pred), "Mismatch in number of labels"
        print(f"{len(Y_test), len(Y_pred)}")
        # Flatten the lists
        # N = len(Y_test)
        # for i in range(N):
        #     print(i)
        #     print (Y_test[i])
        #     print(Y_pred[i])
        Y_test = [label for sentence in Y_test for label in sentence]
        Y_pred = [label for sentence in Y_pred for label in sentence]

        assert len(Y_test) == len(Y_pred), f"Mismatch in number of labels, {len(Y_test), len(Y_pred)}"
        print(f"{len(Y_test), len(Y_pred)}")
        unique_labels = set(Y_test) | set(Y_pred)
        label_counts = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in unique_labels}
        
        for true, pred in zip(Y_test, Y_pred):
            if true == pred:
                label_counts[true]['TP'] += 1
            else:
                label_counts[pred]['FP'] += 1
                label_counts[true]['FN'] += 1
        
        precision, recall, f1_score = {}, {}, {}
        for label, counts in label_counts.items():
            tp, fp, fn = counts['TP'], counts['FP'], counts['FN']
            precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0
        
        all_tp = sum(counts['TP'] for counts in label_counts.values())
        all_fp = sum(counts['FP'] for counts in label_counts.values())
        all_fn = sum(counts['FN'] for counts in label_counts.values())
        print(all_tp, all_fp, all_fn)
        accuracy = sum(counts['TP'] for counts in label_counts.values()) / len(Y_test)
        precision_all = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall_all = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1_score_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if (precision_all + recall_all) > 0 else 0
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_all:.4f}")
        print(f"Recall: {recall_all:.4f}")
        print(f"F1-Score: {f1_score_all:.4f}")
        print("Label-wise Precision, Recall, F1-Score:")
        for label in unique_labels:
            print(f"Label: {label}, Precision: {precision[label]:.4f}, Recall: {recall[label]:.4f}, F1-Score: {f1_score[label]:.4f}")
        
        return accuracy, precision, recall, f1_score

    def eval_from_file(self, filename:str, numlines:int = None)->None:
        
        parsed_data, _, _ = parse(filename=filename, numlines=numlines)

        X_test = parsed_data['Tokens'].tolist()
        pos_test = parsed_data['POS'].tolist()
        Y_test = parsed_data['NER_tags'].tolist()

        N = len(X_test)

        Y_pred = []
        for i in range(0, N):
            y_pred = self.predict_viterbi(X_test[i], pos_test[i])
            Y_pred.append(y_pred)
            print(y_pred,'\n', Y_test[i], '\n\n')

        self.eval(Y_pred=Y_pred, Y_test=Y_test)

def save_crf_model(crf_model, extra:str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"crf_{timestamp}_{extra}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(crf_model, file)
    print(f"CRF model saved to {filename}")

def load_crf_model(filename):
    """Load trained CRF model from file."""
    with open(filename, 'rb') as file:
        crf_model = pickle.load(file)
    print(f"CRF model loaded from {filename}")
    return crf_model

def test():
    print(end_ing('HiAll'))
    print(end_ing('yooooing'))
    c = LinearChainCRF()
    c.fit('data/ner_train.csv', batchsize= 50, numlines=50, show_tqdm=False, maxiter=1)
    # save_crf_model(c, 'em_cond')
    # print(c.train_examples[28389])
    print(c.all_NER_tags)
    print(c.all_POS)
    # print(c.id_funcs)
    print(c.train_examples.columns)
    print(c.pos_dict)
    print(c.ner_dict)
    # print(c.predict_viterbi(['Indian', 'troops', 'shot', 'dead', 'three', 'militants', 'in', 'Doda', 'district', 'Wednesday', '.'], ['JJ', 'NNS', 'VBD', 'JJ', 'CD', 'NNS', 'IN', 'NNP', 'NN', 'NNP', '.']))
    c.eval_from_file('data/ner_test.csv', numlines=10)
if __name__ == "__main__":
    test()
        
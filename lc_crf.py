from utils import *
from typing import DefaultDict, Set, List

class LinearChainCRF:

    def __init__(self):
        self.train_examples = {}
        self.all_NER_tags = set()
        self.all_POS = set()
        self.num_feats = 0

        self.pos_dict = {}
        self.ner_dict = {}
        self.num_ner = 0
        self.num_pos = 0


        self.id_funcs = []

        self.obs_funcs = [
            start_cap,
            end_ing
        ]

        self.obs_feat_names = [
            'start_cap',
            'end_ing'
        ]

        self.feat_funcs = []

        

    # def clean_text(self, text):
    #     return text.translate(str.maketrans('', '', string.punctuation))

    def parse_train(self, filename:str, numlines=None) -> None:
        parsed_data = {}
        i = 0

        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                i+=1
                if (i==numlines): break
                match = re.match(r'Sentence: (\d+)', row['Sentence #'])
                sentence_id = int(match.group(1)) if match else None
                words = row['Sentence'].split()
                pos_tags = eval(row['POS'])  # convert POS tags string to list
                ner_tags = eval(row['Tag'])  # convert NER tags string to list

                if (len(words) != len(pos_tags)) or (len(ner_tags) != len(pos_tags)) or ((len(words) != len(ner_tags))):
                    print(len(words), len(pos_tags), len(ner_tags))
                else:
                    if sentence_id not in parsed_data:
                        parsed_data[sentence_id] = {"Tokens": [], "POS": [], "NER_tags": []}
                        
                    parsed_data[sentence_id]["Tokens"].extend(words)
                    parsed_data[sentence_id]["POS"].extend(pos_tags)
                    parsed_data[sentence_id]["NER_tags"].extend(ner_tags)

                # print(f'{len(ner_tags) == len(pos_tags)}')

                for t in ner_tags:
                    self.all_NER_tags.add(t)

                for pos in pos_tags:
                    self.all_POS.add(pos)

        parsed_data = pd.DataFrame(parsed_data).T

        # self.all_NER_tags = set(parsed_data['Tag'])
        # self.all_POS = set(parsed_data['POS'])
        self.num_ner = len(self.all_NER_tags)
        self.num_pos = len(self.all_POS)

        for idx, t in enumerate(self.all_NER_tags):
            self.ner_dict[t] = idx

        for idx, t in enumerate(self.all_POS):
            self.pos_dict[t] = idx
        
        self.train_examples = parsed_data

    def build_id_funcs(self) -> None:
        for ner_tag in self.all_NER_tags:
            f = lambda t : t == ner_tag
            self.id_funcs.append(f)

    def build_feature_funcs(self) -> None:
        for i in self.id_funcs:
            for q in self.obs_funcs:
                self.feat_funcs.append(lambda y,x : i(y)*q(x))

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

    def emission_score(self, Y:List[str], X:List[str], pos_seq:List[str], t:int, T:int, y:str=None) -> np.float16:
        
        """
        calculates the emission log prob (which includes POS given NER tag, and binary obs functions)
        """

        em = 0.0
        n = self.num_ner
        n2 = n**2
        p = self.num_pos

        if t < T:
            em += self.weights[n2 + 2*n + self.pos_dict[pos_seq[t]]] # for current pos

            # loopify this pls
            curr_token = X[t]
            em += self.weights[n2 + 2*n + p + 0]*self.obs_funcs[0](curr_token)
            em += self.weights[n2 + 2*n + p + 1]*self.obs_funcs[1](curr_token)

        return em
    
    def transition_score(self, Y:List[str], X:List[str], pos_seq:List[str], t:int, T:int, y:str=None, y_:str=None) -> np.float16:
        idx = 0
        n = self.num_ner
        n2 = n**2
        p = self.num_pos

        if t==0:
            return self.weights[self.ner_dict[y] + n2]# BOS -> y
        elif t==T:
            return self.weights[self.ner_dict[y_] + n2 + n]# y_ -> EOS, y is EOS
        else:
            return self.weights[self.ner_dict[y] + n*self.ner_dict[y_]]#for general transition score

    def make_features(self, Y:List[str], X:List[str], pos_seq:List[str], t:int, T:int, y:str=None, y_:str=None):
        """
        Inputs:
            y : str -> current label, y_t (hidden state)
            y_ : str -> prev label, y_(t-1) (hidden state)
            X : List[str] -> obs sequence (tokens list)
            pos_seq : List[str] -> obs sequence (POS list)
            t : int -> timestep
            T : int -> total timesteps (length of sentence)
        """

        try:
            if y == None and t<T:
                y = Y[t]
            if y_ == None and t>0: 
                y_ = Y[t-1]
        except:
            print(T, t, X, Y, len(X), len(pos_seq), len(Y))

        # print(f"Hi {self.num_feats}")
        feats = np.zeros(self.num_feats)
        n = self.num_ner
        n2 = n**2
        p = self.num_pos

        if t==0:
            feats[self.ner_dict[y] + n2] = 1.0 # BOS -> y
        elif t==T:
            feats[self.ner_dict[y_] + n2 + n] = 1.0 # y_ -> EOS, y is EOS
        else:
            feats[self.ner_dict[y] + n*self.ner_dict[y_]] = 1.0 #for transition score
        
        if t < T:
            feats[n2 + 2*n + self.pos_dict[pos_seq[t]]] = 1.0 # for current pos

            # loopify this pls
            curr_token = X[t]
            feats[n2 + 2*n + p + 0] = self.obs_funcs[0](curr_token)
            feats[n2 + 2*n + p + 1] = self.obs_funcs[1](curr_token)

        # print(feats)

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
            score_X_Y += np.dot(self.weights, self.make_features(Y, X, pos_seq, t, T))

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
            j = self.ner_dict[y]
            dp[0][j] = np.dot(self.weights, self.make_features([y], X, pos_seq, 0, T))
            # dp[0][j] = self.weights[n2 + j] + self.

        for t in range(1, T): # goes till T-1, ie before EOS
            for y in self.all_NER_tags:
                j = self.ner_dict[y]

                dp[t][j] = \
                np.logaddexp.reduce([
                    dp[t-1][self.ner_dict[y_]] + self.emission_score([], X, pos_seq, t, T, y=y) +
                    self.transition_score([],X,pos_seq, t, T, y=y, y_=y_)
                    for y_ in self.all_NER_tags
                ])

         #nth = EOS
        dp[T][n] = np.logaddexp.reduce([dp[T-1][self.ner_dict[y_]] + self.transition_score([], X, pos_seq, T, T, y_=y_) for y_ in self.all_NER_tags ])

        return dp[T][n]
        # for y_ in self.all_NER_tags:
        #     j_ = self.ner_dict[y_]
        #     dp[T][n] = np.dot(self.weights, self.make_features([y], X, pos_seq, 0, T)) 
                # dp[t][j] = self.emission_score([], X, pos_seq, t, T, y=y)
                # for y_ in self.all_NER_tags:
                #     j_ = self.ner_dict[y_]
                #     dp[t][j] += dp[t-1][j_] + self.transition_score([],X,pos_seq, t, T, y=y, y_=y_)
    def nll(self, W, X_train:List[List[str]], pos_train:List[List[str]], Y_train:List[List[str]]) -> np.float16:
        self.weights = W
        ll = 0.0
        N = len(X_train)

        r = range(0,N)
        if self.use_tqdm:
            r = tqdm(r)

        for i in r:
            X, pos_seq, Y = X_train[i], pos_train[i], Y_train[i]
            ll += (self.score_seq(X, pos_seq, Y) - self.forward_partition(X, pos_seq))
        return -ll
    
    def callback_function(self, weights):
        """Callback function to print loss during training."""
        loss = self.nll(weights, self.X_train, self.pos_train, self.Y_train)
        print(f"Current NLL Loss: {loss:.4f}")

    def train(self) -> None:

        # initialize weights

        self.num_feats = (self.num_ner)**2 + 2*(self.num_ner) + self.num_pos + len(self.obs_funcs)
        # first set of weights is transition score/prob - [0, n*n-1]
        # second set of weights is transition from BOS to NER tags, and NER tags to EOS - [n*n, n*n + 2n - 1]
        # third set of weights is for current POS tag - [n*n + 2n, n*n + 2n + p - 1]
        # fourth set of wights is for miscellaneous obs funcs - [n*n + 2n + p, n*n + 2n + p + o - 1]
        mu = 0.0
        sigma = 0.01
        self.weights = sigma*np.random.randn(self.num_feats) + np.array([mu]*self.num_feats)

        print(self.num_feats)
        print(self.weights)

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

        print(X_train)
        print(pos_train)
        print(Y_train)

        """ Train using L-BFGS optimizer """
        self.X_train = X_train
        self.Y_train = Y_train
        self.pos_train = pos_train

        result = minimize(self.nll, self.weights, args=(X_train, pos_train, Y_train),
                          method='L-BFGS-B', callback=self.callback_function, options={'disp': True})
        self.weights = result.x  # Update model parameters


    def fit(self, filename:str, numlines:int=None, show_tqdm:bool=False) -> None:
        self.parse_train(filename=filename, numlines=numlines)
        self.build_id_funcs()
        self.build_feature_funcs()
        # self.build_dataframe()
        self.use_tqdm = show_tqdm
        self.train()

    

    def viterbi(self, obs:List[str]) -> List[str]:
        pass


def test():
    print(end_ing('HiAll'))
    print(end_ing('yooooing'))
    c = LinearChainCRF()
    c.fit('data/ner_train.csv', numlines=10, show_tqdm=False)
    # print(c.train_examples[28389])
    print(c.all_NER_tags)
    print(c.all_POS)
    # print(c.id_funcs)
    print(c.train_examples.columns)
    print(c.pos_dict)
    print(c.ner_dict)

if __name__ == "__main__":
    test()
        
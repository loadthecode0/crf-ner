from utils import *
from typing import DefaultDict, Set

class LinearChainCRF:

    def __init__(self):
        self.train_examples = {}
        self.all_tags = set()

        self.id_funcs = []

       
        
        self.obs_funcs = [
            start_cap,
            end_ing
        ]

    # def clean_text(self, text):
    #     return text.translate(str.maketrans('', '', string.punctuation))

    def parse_train(self, filename:str) -> None:
        parsed_data = {}
    
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                match = re.match(r'Sentence: (\d+)', row['Sentence #'])
                sentence_id = int(match.group(1)) if match else None
                words = row['Sentence'].split()
                pos_tags = eval(row['POS'])  # convert POS tags string to list
                ner_tags = eval(row['Tag'])  # convert NER tags string to list

                if sentence_id not in parsed_data:
                    parsed_data[sentence_id] = {"words": [], "pos_tags": [], "ner_tags": []}
                    
                parsed_data[sentence_id]["words"].extend(words)
                parsed_data[sentence_id]["pos_tags"].extend(pos_tags)
                parsed_data[sentence_id]["ner_tags"].extend(ner_tags)

                # print(f'{len(ner_tags) == len(pos_tags)}')

                for t in ner_tags:
                    self.all_tags.add(t)
        
        self.train_examples = parsed_data

    def build_id_funcs(self) -> None:

        for ner_tag in self.all_tags:
            f = lambda t : t == ner_tag
            self.id_funcs.append(f)

    def train(self) -> None:
        pass

    def fit(self, filename:str) -> None:
        self.parse_train(filename=filename)
        self.build_id_funcs()
        self.train()


def test():
    print(end_ing('HiAll'))
    print(end_ing('yooooing'))
    c = LinearChainCRF()
    c.fit('data/ner_train.csv')
    print(c.train_examples[28389])
    print(c.all_tags)
    print(c.id_funcs)

if __name__ == "__main__":
    test()
        
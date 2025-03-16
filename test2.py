from lc_crf import LinearChainCRF

c2 = LinearChainCRF()

c2.fit('data/ner_train.csv', batchsize= 100, numlines=1000, show_tqdm=False, maxiter=20, train=True)
c2.eval_from_file('data/ner_test.csv', numlines=30)
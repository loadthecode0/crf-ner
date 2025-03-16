from lc_crf import LinearChainCRF

c2 = LinearChainCRF()

c2.fit('data/ner_train.csv', batchsize= 50, numlines=10, show_tqdm=False, maxiter=25, train=True)
c2.eval_from_file('data/ner_test.csv', numlines=10)
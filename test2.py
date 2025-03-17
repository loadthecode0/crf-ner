from lc_crf import LinearChainCRF, save_crf_model, load_crf_model

c2 = LinearChainCRF()

c2.fit('data/ner_train.csv', batchsize=50, numlines=50, show_tqdm=False, maxiter=20, train=True)
# save_crf_model(c2, 'em_cond_50egs')
c2.eval_from_file('data/ner_test.csv', numlines=10)
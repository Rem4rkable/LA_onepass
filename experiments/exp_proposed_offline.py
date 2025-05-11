import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import gete2wlandw2el,getaccuracy
import time
from results_summary import OfflineResultsSummary
from LA_methods.proposed.proposed_method import one_pass,two_pass
random.seed(1)

if __name__ == '__main__':
    datasets = [
        ('crowdscale2013/sentiment', 'senti'),
        ('crowdscale2013/fact_eval', 'fact'),
        ('active-crowd-toolkit/CF', 'CF'),
        ('active-crowd-toolkit/CF_amt', 'CF_amt'),
        ('active-crowd-toolkit/MS', 'MS'),
        ('crowd_truth_inference/s4_Face Sentiment Identification', 'face'),
        ('crowd_truth_inference/s5_AdultContent', 'adult'),
        ('SpectralMethodsMeetEM/dog', 'dog'),
        ('SpectralMethodsMeetEM/web', 'web'),
        ('mill','mill'),

        ('active-crowd-toolkit/SP', 'SP'),
        ('active-crowd-toolkit/SP_amt', 'SP_amt'),
        ('active-crowd-toolkit/ZenCrowd_all', 'ZC_all'),
        ('active-crowd-toolkit/ZenCrowd_in', 'ZC_in'),
        ('active-crowd-toolkit/ZenCrowd_us', 'ZC_us'),
        ('crowd_truth_inference/d_jn-product', 'product'),
        ('crowd_truth_inference/d_sentiment', 'tweet'),
        ('SpectralMethodsMeetEM/bluebird', 'bird'),
        ('SpectralMethodsMeetEM/rte', 'rte'),
        ('SpectralMethodsMeetEM/trec', 'trec'),
    ]
    
    datasets = [
        ('rbm/mniste_cs_test_dataset','MnistE'),
        ('rbm/condind_test_dataset','CondInd'),
        ('rbm/hs3_test_dataset','HS3'),
        ('rbm/mixedclf_test_dataset','MixedClf'),
        ('rbm/nnt_test_dataset','NeuralNet'),
        ('rbm/c2_boosting_test_dataset','C-Boosting'),
        ('rbm/c2_complement_test_dataset','C-Complement'),
        
    ]
    datasets = [
        #('deem/mniste_train_dataset','MnistE'),
        #('deem/hs3_train_dataset','HS3'),
        #('deem/petfinder_train_dataset','Petfinder'),
        ('deem/mboosting_train_dataset','MBoosting'),
        ('deem/mcomplement_train_dataset','MComplement'),
    ]
    datasets = [
    #('deem/hs3_train_dataset', 'HS3'),
    #('deem/mniste_train_dataset', 'MnistE'),
    #('deem/petfinder_train_dataset', 'Petfinder'),
    #('deem/artificial-characters_train_dataset', 'ArtificialCharacters'),
    #('deem/csgo_train_dataset', 'CSGO'),
    #('deem/eye_movements_train_dataset', 'EyeMovements'),
    #('deem/GesturePhaseSegmentationProcessed_train_dataset', 'GesturePhaseSegmentation'),
    #('deem/mboosting_train_dataset', 'MBoosting'),
    #('deem/mcomplement_train_dataset', 'MComplement'),
    #('deem/microaggregation2_train_dataset', 'Microaggregation2'),
    #('deem/pendigits_train_dataset', 'Pendigits'),
    #('deem/volcanoes-b2_train_dataset', 'VolcanoesB2'),
    ('deem/tree3k_train_dataset', 'Tree3k')
    ]

    results_onepass = OfflineResultsSummary('onepass')
    results_twopass = OfflineResultsSummary('twopass')

    round = 10
    for dataset, abbrev in datasets:
        print(dataset, abbrev)
        label_path = "data/"+dataset+"/label.csv"
        truth_path = "data/"+dataset+"/truth.csv"

        e2wl, w2el, label_set = gete2wlandw2el(label_path)
        for r in range(round):
            t1 = time.time()
            one_pass_truths, a = one_pass(e2wl, w2el, label_set, alpha=2, beta=2)
            t2 = time.time()

            one_pass_acc = getaccuracy(truth_path, one_pass_truths)


            t3 = time.time()
            two_pass_truths = two_pass(e2wl, a, label_set)
            t4 = time.time()
            two_pass_acc = getaccuracy(truth_path, two_pass_truths)

            results_onepass.add(abbrev, one_pass_acc, t2 - t1, 1)
            results_twopass.add(abbrev, two_pass_acc, t2 - t1 + t4 - t3, 2)


    print(results_onepass.to_dataframe_mean())
    print(results_twopass.to_dataframe_mean())


# python experiments/exp_proposed_offline.py
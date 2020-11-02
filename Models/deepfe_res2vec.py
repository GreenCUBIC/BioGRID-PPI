# -*- coding: utf-8 -*-

"""
    ---------- Original Work this file is based on: ----------
    Title: 'An integration of deep learning with feature embedding for protein–protein interaction prediction'
    Authors: Yao Y, Du X, Diao Y, Zhu H.
    Source: PeerJ, 7, e7126. https://doi.org/10.7717/peerj.7126
    Year: 2019
    Month: 06
    DOI: http://dx.doi.org/10.7717/peerj.7126
    git: https://github.com/xal2019/DeepFE-PPI
    
    ---------- This file ----------
    This deepfe_res2vec.py file is a modification from the original git files 5cv_11188.py and swiss_Res2vec_val_11188.py
    Main modifications include a change of command-line argument usage for execution and a choice of cross-validation 
    or a single train/test split. Prediction probabilities of each interaction in test data are also saved to file.
    Author: Eric Arezza
    Last Updated: October 28, 2020
    
    Description:
        Res2evc embedding for sequence representation in deep learning approach to binary classification of protein-protein interaction prediction.
"""

__all__ = ['averagenum',
           'max_min_avg_length',
           'merged_DBN',
           'token',
           'padding_J',
           'protein_representation',
           'read_Data',
           'read_proteinData',
           'get_dataset',
           'mkdir',
           'get_traintest_split',
           'get_crossvalidation_splits',
           'get_test_results'
           ]

import os, argparse
from time import time
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import numpy as np
from keras.layers.core import Dense, Dropout, Merge
#from keras.layers.merge import Concatenate
from sklearn.preprocessing import StandardScaler
import utils.tools as utils
from keras.regularizers import l2
from gensim.models import Word2Vec
import copy
import h5py
from sklearn.model_selection import StratifiedKFold, KFold
from keras import backend as K
import tensorflow as tf
import pandas as pd
from keras.optimizers import SGD
from datetime import datetime
import psutil

# Description of command-line usage
describe_help = 'python deepfe_res2vec.py trainFiles/ testFiles/'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('train', help='Path to file containing binary protein interactions for training (.tsv)', type=str, nargs=1)
parser.add_argument('test', help='Path to file containing binary protein interactions for testing (.tsv)', type=str, nargs=1)
parser.add_argument('-s', '--size', help='Size (int)', type=int, nargs=1, required=False)
parser.add_argument('-w', '--window', help='Window size (int)', type=int, nargs=1, required=False)
parser.add_argument('-l', '--length', help='Max length (int)', type=int, nargs=1, required=False)
parser.add_argument('-b', '--batch', help='Batch size (int)', type=int, nargs=1, required=False)
parser.add_argument('-e', '--epochs', help='Epochs (int)', type=int, nargs=1, required=False)
parser.add_argument('-r','--results', help='Path to file to store results', type=str, nargs=1, required=False)
args = parser.parse_args()

if args.window is None:
    window = 4
else:
    window = args.window[0]
if args.batch is None:
    batch_size = 256
else:
    batch_size = args.batch[0]
if args.epochs is None:
    n_epochs = 50
else:
    n_epochs = args.epochs[0]
if args.size is None:
    size = 20
else:
    size = args.size[0]
if args.length is None:
    maxlen = 850
else:
    maxlen = args.length[0]
if args.results is None:
    #rst_file = os.getcwd()+'/results/'+datetime.now().strftime("%d-%m-%Y/")+datetime.now().strftime("%H-%M-%S-results.txt")    
    rst_file = os.getcwd()+'/results_'+datetime.now().strftime("%d-%m-%Y_")+datetime.now().strftime("%H-%M-%S.txt")
else:
    rst_file = args.results

TRAIN_PATH = args.train[0]
TEST_PATH = args.test[0]
CROSS_VALIDATE = False
if TRAIN_PATH == TEST_PATH:
    CROSS_VALIDATE = True

print("\n---Using the following---\nTraining Files: {}\nTesting Files: {}".format(TRAIN_PATH, TEST_PATH))
print("Size: {}\nWindow: {}\nLength: {}\nBatch: {}\nEpochs: {}\n".format(size, window, maxlen, batch_size, n_epochs))

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)
   
def max_min_avg_length(seq):
    length = []
    for string in seq:
        length.append(len(string))   
    maxNum = max(length) #maxNum = 5
    minNum = min(length) #minNum = 1
    
    avg = averagenum(length)
    
    print('The longest length of protein is: '+str(maxNum))
    print('The shortest length of protein is: '+str(minNum))
    print('The avgest length of protein is: '+str(avg))

def merged_DBN(sequence_len):
    # left model
    model_left = Sequential()
    model_left.add(Dense(2048, input_dim=sequence_len ,activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
   
    model_left.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    
    # right model
    model_right = Sequential()
    model_right.add(Dense(2048,input_dim=sequence_len,activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
     
    model_right.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    # together
    merged = Merge([model_left, model_right])
      
    model = Sequential()
    model.add(merged)
    model.add(Dense(8, activation='relu',W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.summary()
    
    return model
    
def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])
        token_dataset.append(seq)  
    return  token_dataset
    
def padding_J(protein,maxlen):           
    padded_protein = copy.deepcopy(protein)   
    for i in range(len(padded_protein)):
        if len(padded_protein[i])<maxlen:
            for j in range(len(padded_protein[i]),maxlen):
                padded_protein[i].append('J')
    return padded_protein  
    
def protein_representation(wv,tokened_seq_protein,maxlen,size):  
    represented_protein  = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j]=='J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))    
    return represented_protein
    
def read_Data(file_name):
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                seq.append(line.split('\n')[0])
            i = i+1       
    return seq   

def read_proteinData(file_name):
    seq = []
    with open(file_name, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==0:
                seq.append(line.split('\n')[0].strip('>'))
            i = i+1       
    return seq   

def get_dataset(wv,  maxlen,size, files, data='train'):
    
    if data == 'train':
        path = TRAIN_PATH
    else:
        path = TEST_PATH
    if path[-1] != '/':
        path += '/'
    
    for f in files:
        if 'positive' in f.lower() or '_pos_' in f.lower():
            if 'proteina' in f.lower() or 'protein_a' in f.lower():
                file_1 = path + f
            if 'proteinb' in f.lower() or 'protein_b' in f.lower():
                file_2 = path + f
        if 'negative' in f.lower() or '_neg_' in f.lower():
            if 'proteina' in f.lower() or 'protein_a' in f.lower():
                file_3 = path + f
            if 'proteinb' in f.lower() or 'protein_b' in f.lower():
                file_4 = path + f

    # positive seq protein A
    pos_seq_protein_A = read_Data(file_1)
    pos_seq_protein_B = read_Data(file_2)
    neg_seq_protein_A = read_Data(file_3)
    neg_seq_protein_B = read_Data(file_4)
    # put pos and neg together
    pos_neg_seq_protein_A = copy.deepcopy(pos_seq_protein_A)   
    pos_neg_seq_protein_A.extend(neg_seq_protein_A)
    pos_neg_seq_protein_B = copy.deepcopy(pos_seq_protein_B)   
    pos_neg_seq_protein_B.extend(neg_seq_protein_B)
    
    # Sequence stats
    seq = []
    seq.extend(pos_neg_seq_protein_A)
    seq.extend(pos_neg_seq_protein_B)
    max_min_avg_length(seq)
    
    # token
    token_pos_neg_seq_protein_A = token(pos_neg_seq_protein_A)
    token_pos_neg_seq_protein_B = token(pos_neg_seq_protein_B)
    # padding
    tokened_token_pos_neg_seq_protein_A = padding_J(token_pos_neg_seq_protein_A, maxlen)
    tokened_token_pos_neg_seq_protein_B = padding_J(token_pos_neg_seq_protein_B, maxlen)
    # protein reprsentation
    feature_protein_A  = protein_representation(wv, tokened_token_pos_neg_seq_protein_A, maxlen,size)
    feature_protein_B  = protein_representation(wv, tokened_token_pos_neg_seq_protein_B, maxlen,size)
    feature_protein_AB = np.hstack((np.array(feature_protein_A), np.array(feature_protein_B)))
    #  create label
    label = np.ones(len(feature_protein_A))
    label[len(feature_protein_AB)//2:] = 0
    
    # Get protein pairs by protein ID
    pairs = []
    pos_protein_A = read_proteinData(file_1)
    pos_protein_B = read_proteinData(file_2)
    neg_protein_A = read_proteinData(file_3)
    neg_protein_B = read_proteinData(file_4)
    # put pos and neg pairs together
    pos_neg_protein_A = copy.deepcopy(pos_protein_A)   
    pos_neg_protein_A.extend(neg_protein_A)
    pos_neg_protein_B = copy.deepcopy(pos_protein_B)   
    pos_neg_protein_B.extend(neg_protein_B)
    
    for p in range(0, len(pos_neg_protein_A)):
        pairs.append([pos_neg_protein_A[p], pos_neg_protein_B[p]])
   
    return pairs, feature_protein_AB, label
    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                 
        os.makedirs(path)           
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def get_traintest_split(class_labels, train_length):
    train = class_labels[:train_length]
    test = class_labels[train_length:]
    train = []
    test = []
    for i in range(0, train_length):
        train.append(i)
    for j in range(train_length, len(class_labels)):
        test.append(j)

    return [(np.asarray(train), np.asarray(test))]

def get_crossvalidation_splits(training, class_labels, nsplits=5):
    
    #skf = StratifiedKFold(n_splits=nsplits, random_state=20181031, shuffle=True)
    #train_test = skf.split(training, class_labels)
    
    # KFold is used since datasets are assumed to be balanced positive/negative
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=10312020)
    tries = 5
    cur = 0
    train_test = []
    for train, test in kf.split(training, class_labels):
        if np.sum(class_labels[train], 0)[0] > 0.8 * len(train) or np.sum(class_labels[train], 0)[0] < 0.2 * len(train):
            continue
        train_test.append((train, test))
        cur += 1
        if cur >= tries:
            break
    
    return train_test

def get_test_results(raw_data, test_indices, predictions, class_labels):
    # raw_data provides train+test pairs
    # test_indices provides indices of test data for raw_data pairs
    # predictions contains model predictions of each pair
    # class_labels provides labels for interaction pairs (1 or 0)
    prob_results = []
    for ppi in range(0, len(test_indices)):
        proteinA = str(raw_data[test_indices[ppi]][0])
        proteinB= str(raw_data[test_indices[ppi]][1])
        
        # Get index for correctly predicted class label
        #true_interact_index = np.argmax(class_labels[test_indices[ppi]])
        # Get indices for positive ('1') class label
        pos_interact_index = 1
        # Add prediction probability of interaction being positive
        prob_pos_interaction = predictions[ppi][pos_interact_index]
        prob_results.append([proteinA + ' ' + proteinB + ' ' + str(prob_pos_interaction)])
        
    return np.asarray(prob_results, dtype=str)

'''
    The following functions are used to build res2vec representation
''' 
def get_res2vec_data(files):
    sequences = []
    for file in files:
        fasta = []
        with open(file, 'r') as fp:
            protein = ''
            for line in fp:
                if line.startswith('>'):
                    fasta.append(protein)
                    protein = ''
                elif line.startswith('>') == False:
                    protein = protein+line.strip()
            sequences.extend(list(set(fasta[1:])))
    sequences = list(set(sequences))
    return sequences

def getMemorystate():   
    phymem = psutil.virtual_memory()   
    line = "Memory: %5s%% %6s/%s"%(phymem.percent,
            str(int(phymem.used/1024/1024))+"M",
            str(int(phymem.total/1024/1024))+"M") 
    return line 

def res2vec(size, window, maxlen, files):
    data = token(get_res2vec_data(files))   # token
    t_start = time()
    model = Word2Vec(data, size = size, min_count = 0, sg =1, window = window)
    print('memInfo_wv ' + getMemorystate())
    print('Word2Vec model is created')
    sg = 'wv_' + TRAIN_PATH.split('/')[-2] + '_size_'+str(size)+'_window_'+str(window) 
    print('Time to create Word2Vec model ('+sg+'):', time() - t_start)
    model.save('model/word2vec/'+sg+'.model')
    
    # load dictionary
    model_wv = Word2Vec.load('model/word2vec/'+sg+'.model')
    
    return model_wv

#%%  
if __name__ == "__main__": 
    
    try:
        train_files = os.listdir(path=TRAIN_PATH)
        print('Using training files:\n', train_files)
        test_files = os.listdir(path=TEST_PATH)
        print('Using testing files:\n', test_files)
        if len(train_files) < 4 or len(test_files) < 4:
            print("Check positive/negative proteinA/B files...")
            exit()
    except Exception as e:
        print(e, "\nPlease provide path to files, for example: 'python deepfe_res2vec.py train/ test/'")
        exit()
    
    seq_files = []
    if TRAIN_PATH[-1] != '/':
        path = TRAIN_PATH + '/'
    else:
        path = TRAIN_PATH
    for f in range(0, len(train_files)):
        seq_files.append(path + train_files[f])
    # load dictionary
    #model_wv = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')
    # make dictionary
    model_wv = res2vec(size, window, maxlen, seq_files)
    
    sequence_len = size*maxlen
                       
    # get training data 
    t_start = time() 
    
    if not CROSS_VALIDATE:
        # Process training data
        raw_pairs_train, train_fea_protein_AB, train_label = get_dataset(model_wv.wv, maxlen, size, train_files, data='train')
        Y_train = utils.to_categorical(train_label)
        # Process testing data
        raw_pairs_test, test_fea_protein_AB, test_label = get_dataset(model_wv.wv, maxlen, size, test_files, data='test')
        Y_test = utils.to_categorical(test_label)
        # Combine to common variable, but keep train/test split
        raw_pairs = raw_pairs_train + raw_pairs_test
        fea_protein_AB = np.vstack((train_fea_protein_AB, test_fea_protein_AB))
        scaler = StandardScaler().fit(fea_protein_AB)
        fea_protein_AB = scaler.transform(fea_protein_AB)
        Y = np.concatenate((Y_train, Y_test))
        train_test = get_traintest_split(Y, Y_train.shape[0])
    else:
        raw_pairs, fea_protein_AB, train_label = get_dataset(model_wv.wv, maxlen, size, train_files, data='train')  
        #scaler
        scaler = StandardScaler().fit(fea_protein_AB)
        scaled_fea_protein_AB = scaler.transform(fea_protein_AB)
        Y = utils.to_categorical(train_label)
        train_test = get_crossvalidation_splits(scaled_fea_protein_AB, Y, nsplits=5)
        '''
        print('FEA_PROTEIN_AB:', len(fea_protein_AB))
        print(fea_protein_AB)
        print('TRAIN_LABEL:', len(train_label))
        print(train_label)
        print('Y:', len(Y))
        print(Y)
        print('TRAIN/TEST SPLIT:', len(train_test))
        print(train_test)
        print('SCALED_FEA_PROTEIN_AB:', len(scaled_fea_protein_AB))
        print(scaled_fea_protein_AB)
        fea_protein_AB = scaled_fea_protein_AB
        '''
        '''
        # If using pre-trained
        h5_file = h5py.File('dataset/11188/wv_swissProt_size_'+str(size)+'_window_'+str(window)+'_maxlen_'+str(maxlen)+'.h5','r')
        h5_fea_protein_AB =  h5_file['trainset_x'][:]
        h5_train_label = h5_file['trainset_y'][:]
        h5_file.close()
        print('dataset is loaded')
        h5_Y = utils.to_categorical(train_label)
        #h5_train_test = get_crossvalidation_splits(h5_fea_protein_AB, h5_train_label, nsplits=5)
        print('\nH5_FEA_PROTEIN_AB:', len(h5_fea_protein_AB))
        print(h5_fea_protein_AB)
        print('H5_TRAIN_LABEL:', len(h5_train_label))
        print(h5_train_label)
        print('H5_Y:', len(h5_Y))
        print(h5_Y)
        #print('H5_TRAIN/TEST SPLIT:', len(h5_train_test))
        #print(h5_train_test)
        '''
        
    scores = []
    i = 0
    for (train_index, test_index) in train_test:

        # Get features
        X_train_left = np.array(fea_protein_AB[train_index][:,0:sequence_len])
        X_train_right = np.array(fea_protein_AB[train_index][:,sequence_len:sequence_len*2])
        X_test_left = np.array(fea_protein_AB[test_index][:,0:sequence_len])
        X_test_right = np.array(fea_protein_AB[test_index][:,sequence_len:sequence_len*2])

        # Get labels
        y_train = Y[train_index]
        y_test = Y[test_index]
        
        # Build model
        model =  merged_DBN(sequence_len)  
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['precision'])
        # feed data into model
        hist = model.fit([X_train_left, X_train_right], y_train,
                         batch_size = batch_size,
                         nb_epoch = n_epochs,
                         verbose = 1)
        # Make predictions
        predictions = model.predict([X_test_left, X_test_right]) 
        
        # Save predictions to separate file
        if not CROSS_VALIDATE:
            # Save interaction probability results
            prob_results = get_test_results(raw_pairs, test_index, predictions, Y)
            np.savetxt('results_' + str(i) + '_' + os.path.split(TEST_PATH)[0].split('/')[-1] + '.txt', prob_results, fmt='%s', delimiter='\n')
        else:
            # Save interaction probability results
            prob_results = get_test_results(raw_pairs, test_index, predictions, Y)
            np.savetxt('results_' + os.path.split(TEST_PATH)[0].split('/')[-1] + '_test_' + str(i) + '.txt', prob_results, fmt='%s', delimiter='\n')

        auc_test = roc_auc_score(y_test[:,1], predictions[:,1])
        pr_test = average_precision_score(y_test[:,1], predictions[:,1])
        
        label_predict_test = utils.categorical_probas_to_classes(predictions)  
        tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= utils.calculate_performace(len(label_predict_test), label_predict_test, y_test[:,1])
        print(' ===========  test:'+str(i))
        print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp_test,fp_test,tn_test,fn_test))
        print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
              % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test, f1_score_test))
        print('\tauc=%0.4f,pr=%0.4f'%(auc_test,pr_test))
        scores.append([accuracy_test,precision_test, recall_test,specificity_test, MCC_test, f1_score_test, auc_test,pr_test]) 
        
        i=i+1
        K.clear_session()
        tf.reset_default_graph()
        
        sc= pd.DataFrame(scores)   
#        sc.to_csv(result_dir+swm+be+'.csv')   
        scores_array = np.array(scores)
        print(("accuracy=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100)))
        print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100)))
        print("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
        print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
        print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
        print("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
        print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
        print("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
        
        print(time() - t_start)

        with open(rst_file,'w') as f:
           f.write('accuracy=%.2f%% (+/- %.2f%%)' % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100))
           f.write('\n')
           f.write("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100))
           f.write('\n')
           f.write("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
           f.write('\n')
           f.write("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
           f.write('\n')
           f.write("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
           f.write('\n')
           f.write("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
           f.write('\n')
           f.write("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
           f.write('\n')
           f.write("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
           f.write('\n')

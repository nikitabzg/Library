import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from mne.io import read_raw_eeglab, concatenate_raws, read_epochs_eeglab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from mne.channels import read_montage
from mne.event import find_events
from sklearn import metrics
import random
import os
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM, TSclassifier
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.spatialfilters import Xdawn, CSP
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP as MNE_CSP
from Library.tools import DownSampler, EpochsVectorizer, CospBoostingClassifier
from pyriemann.channelselection import ElectrodeSelection

"""
All the used pipelines
"""

def create_folder(k, number_of_subjects, shuffle=True, random_state = 42):
    """
    Creates the k-folds folder for the data
    """
    if k > number_of_subjects:
        folder = LeaveOneOut()
    if k == number_of_subjects:
        folder = LeaveOneOut()
    else : 
        folder = KFold(k, shuffle=shuffle, random_state=random_state)
    return folder


def estimate_covariance_matrices(raw_dataset):
    """
    Creates the simple covariance matrices for the raw data
    """
    covariance_matrices = []
    for subject in raw_dataset:
        covariance_matrices.append(Covariances("oas").transform(subject))
    return np.asarray(covariance_matrices)


def compute_power_spectral_density(windowed_signal, psd_freqs, sampling_freq, overlap):
    """
    Compute the PSD of each 32 electrodes and form a binned spectrogram of 5 frequency bands
    Return the log_10 on the 32 spectrograms 
    """
    # Windowed signal of shape (9 x 513)
    n_electrodes = windowed_signal.shape[0]
    ret = np.empty((psd_freqs.shape[0], n_electrodes), dtype=np.float32)

    # Welch parameters
    sliding_window = sampling_freq
    n_overlap = int(sliding_window * overlap)

    # compute psd using Welch method
   
    freqs, power = welch(windowed_signal, fs=sampling_freq,
        nperseg=sliding_window, noverlap=n_overlap)

    for i, psd_freq in enumerate(psd_freqs):
        tmp = (freqs >= psd_freq[0]) & (freqs < psd_freq[1])
        ret[i] = power[:, tmp].mean(1)
    return np.log(ret)

def apply_PSD(raw_dataset, psd_freqs=np.array([[1, 4], [4, 8], [8, 15], [15, 20], [30, 40]]), sampling_freq=512, overlap=0.25):
    """
    Applies the power spectral density method to a dataset and returns the computed matrix.
    """
    psd_matrix = []
    for person in raw_dataset:
        psd_matrix.append([])
        for epoch in person:
            signal_psd = compute_power_spectral_density(epoch, psd_freqs, 512, 0.25)
            signal_psd = np.ndarray.flatten(signal_psd)
            psd_matrix[-1].append(signal_psd)

    psd_matrix = np.asarray(psd_matrix)

    return psd_matrix

class Pipeline_catalogue:
    def __init__(self):

        self.CONST_MAX_ELECTRODES = 30

        self.catalogue = {}

        # ERPs models
        """
        self.catalogue['XdawnCovTSLR'] = make_pipeline(XdawnCovariances(3, estimator='oas'),
                                        TangentSpace('riemann'),
                                        LogisticRegression('l2'))

        self.catalogue['XdawnCov'] =  make_pipeline(XdawnCovariances(3, estimator='oas'),
                                        MDM(metric=dict(mean='riemann', distance='riemann')))
        
        self.catalogue['Xdawn'] = make_pipeline(Xdawn(12, estimator='oas'),
                                        DownSampler(5),
                                        EpochsVectorizer(),
                                        LogisticRegression('l2'))
        self.catalogue['CSP'] = make_pipeline(Xdawn(12, estimator='oas'),
                                      MNE_CSP(8),
                                      LogisticRegression('l2'))
        """
        self.catalogue['CSP2'] = make_pipeline(ElectrodeSelection(10),
                                      MNE_CSP(8),
                                      LogisticRegression('l2'))
        
        # Induced activity models
       
        self.catalogue['cov'] = make_pipeline(ElectrodeSelection(10),
                            TangentSpace('riemann'),
                            LogisticRegression('l1'))
        """
        self.catalogue['Cosp'] = make_pipeline(CospCovariances(fs=1000, window=32, overlap=0.95, fmax=300, fmin=1),
                                           CospBoostingClassifier(baseclf))

        self.catalogue['HankelCov'] = make_pipeline(DownSampler(2),
                                                HankelCovariances(delays=[2, 4, 8, 12, 16], estimator='oas'),
                                                TangentSpace('logeuclid'),
                                                LogisticRegression('l1'))

        self.catalogue['CSSP'] = make_pipeline(HankelCovariances(delays=[2, 4, 8, 12, 16], estimator='oas'),
                                           CSP(30),
                                           LogisticRegression('l1'))

        # additionnal pipelines

        self.catalogue['PSD'] = LogisticRegression()

        self.catalogue['MDM'] = MDM(metric=dict(mean='riemann', distance='riemann'))
        """

        self.scores_inter = []
        self.confusion_matrices_inter = []
        self.predictions_probabilities_inter = []
        self.precisions_inter = []
        self.recalls_inter = []

        self.scores_intra = []
        self.confusion_matrices_intra = []
        self.predictions_probabilities_intra = []
        self.precisions_intra = []
        self.recalls_intra = []

    def modify_add_pipeline(self, name, pipeline):
        """
        Allows the user to add their own pipeline to the available catalogue, or to modify a pipeline (for example, using their own
        hypervalues for a pipeline, such as the number of electrodes to select, of CSP filters, of xDAWN filters, etc...).
        """
        self.catalogue[name] = pipeline

    def set_electrodes_number(self, dataset, labels, k=5, shuffle=True, random_state = 42):
        """
        Finds the number of electrodes to selects that is most likely to yield good results. 
        If the available number of electrodes is relatively small, all numbers of electrodes to keep are tested one by one.
        If the available number of electrodes is relatively big, multiples of 10 are tested.
        """
        dataset = estimate_covariance_matrices(dataset)
        number_of_electrodes = dataset[0].shape[1]
        total_epochs = np.sum([dataset[i].shape[0] for i in range(dataset.shape[0])])

        folder = create_folder(k, dataset.shape[0], shuffle=shuffle, random_state = random_state)
        scores_CSP2 = []
        scores_cov = []
        indices = []
        if (number_of_electrodes >= self.CONST_MAX_ELECTRODES):
            step = 10
        else : 
            step = 1
        print(step)
        for i in range(1,number_of_electrodes+1,step):
            print(i)
            CSP2 = make_pipeline(ElectrodeSelection(i),
                                  MNE_CSP(8),
                                  LogisticRegression('l2'))
            
            cov = make_pipeline(ElectrodeSelection(i),
                        TangentSpace('riemann'),
                        LogisticRegression('l1'))
            score_CSP2 = 0
            score_cov = 0

            for train_index, test_index in folder.split(dataset):
                x_train = np.asarray([dataset[train_index[i]] for i in range(len(train_index))])
                x_train = np.concatenate(x_train)
                x_test = np.asarray([dataset[test_index[i]] for i in range(len(test_index))])
                x_test = np.concatenate(x_test)

                y_train = np.asarray([labels[train_index[i]] for i in range(len(train_index))])
                y_train = np.concatenate(y_train)
                y_test = np.asarray([labels[test_index[i]] for i in range(len(test_index))])
                y_test = np.concatenate(y_test)
                
                CSP2.fit(x_train, y_train)                
                preds_CSP2 = CSP2.predict(x_test)

                cov.fit(x_train, y_train)                
                preds_cov = cov.predict(x_test)

                for j in range(y_test.shape[0]):
                    if (preds_CSP2[j] == y_test[j]) : 
                        score_CSP2 += 1
                    if (preds_cov[j] == y_test[j]) : 
                        score_cov += 1  

            scores_CSP2.append(score_CSP2/total_epochs)
            scores_cov.append(score_cov/total_epochs)  
            indices.append(i)

        max_CSP2 = indices[np.argmax(scores_CSP2)]
        max_cov = indices[np.argmax(scores_cov)]

        self.catalogue["CSP2"] = make_pipeline(ElectrodeSelection(max_CSP2, metric=dict(mean='logeuclid',distance='riemann')),
                                  MNE_CSP(8),
                                  LogisticRegression('l2'))

        self.catalogue["cov"] = make_pipeline(ElectrodeSelection(max_cov, metric=dict(mean='logeuclid',distance='riemann')),
                        TangentSpace('riemann'),
                        LogisticRegression('l1'))

        self.plot_classification_by_electrodes(indices, [scores_CSP2, scores_cov], ["CSP2","cov"])



    def launch_all_pipelines_intersubject(self, dataset, labels, k=5, shuffle=True, random_state = 42):
        """
        Launches every pipeline one by one on splitted data and, at the end, shows each pipeline's classification results.
        This is done by leaving a subset of subjects out to be tested on, and fitting the classifiers on the other subjects.
        """
        folder = create_folder(k, dataset.shape[0], shuffle=shuffle, random_state = random_state)

        pipelines = list(self.catalogue.keys())
        pipelines.sort()
        for pipeline_name in pipelines :
            pipeline = self.catalogue[pipeline_name]
            score = 0
            precision = 0
            recall = 0
            confusion_matrix = [[0 for j in range(2)] for i in range(2)] 
            for train_index, test_index in folder.split(dataset):
                x_train = np.asarray([dataset[train_index[i]] for i in range(len(train_index))])
                x_train = np.concatenate(x_train)
                x_test = np.asarray([dataset[test_index[i]] for i in range(len(test_index))])
                x_test = np.concatenate(x_test)

                y_train = np.asarray([labels[train_index[i]] for i in range(len(train_index))])
                y_train = np.concatenate(y_train)
                y_test = np.asarray([labels[test_index[i]] for i in range(len(test_index))])
                y_test = np.concatenate(y_test)

                pipeline.fit(x_train, y_train)                
                preds_proba = pipeline.predict_proba(x_test)
                preds = np.argmax(preds_proba, axis=1)
                for pred_idx in range(preds.shape[0]) :
                    confusion_matrix[y_test[pred_idx]][preds[pred_idx]] += 1
            score = (confusion_matrix[0][0] + confusion_matrix[1][1])/np.sum(confusion_matrix)
            precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
            recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
            self.scores_inter.append(score)
            self.confusion_matrices_inter.append(confusion_matrix)
            self.predictions_probabilities_inter.append(preds_proba)
            self.precisions_inter.append(precision)
            self.recalls_inter.append(recall)
            print(pipeline_name,":")    
            print("Score : ", score)
            print("Precision : ", precision)
            print("Recalll : ", recall)
            print("Confusion matrix : ", confusion_matrix)
            print("_________________")



    def launch_all_pipelines_intrasubject(self, dataset, labels, k, shuffle=True, random_state = 42):
        """
        Launches every pipeline one by one on splitted data and, at the end, shows each pipeline's classification results.
        The learning and the testing phase are done one subject at the time : one subject is taken out and their data are folded
        in k folds. 1 fold is kept for testing while the rest are used for learning/fitting the pipeline.
        """
        folder = create_folder(k, dataset.shape[0], shuffle=shuffle, random_state = random_state)
        loo = LeaveOneOut()
        pipelines = list(self.catalogue.keys())
        pipelines.sort()
        for pipeline_name in pipelines :
            pipeline = self.catalogue[pipeline_name]
            score = 0
            precision = 0
            recall = 0
            confusion_matrix = [[0 for j in range(2)] for i in range(2)] 
            for train_index, test_index in loo.split(dataset):

                x_train = np.asarray(dataset[test_index[0]])
                y_train = np.asarray(labels[test_index[0]])


                for train, test in folder.split(x_train, y_train):
                    correct_subject = 0
                    incorrect_subject = 0
                    pipeline.fit(x_train[train], y_train[train])
                    preds_proba = pipeline.predict_proba(x_train[test])
                    preds = np.argmax(preds_proba, axis=1)
                    for prediction_idx in range(preds.shape[0]): 
                       confusion_matrix[y_train[test][prediction_idx]][preds[prediction_idx]] += 1
                    #input(preds_proba.shape)

            score = (confusion_matrix[0][0] + confusion_matrix[1][1])/np.sum(confusion_matrix)
            precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
            recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
            self.scores_intra.append(score)
            self.confusion_matrices_intra.append(confusion_matrix)
            self.predictions_probabilities_intra.append(preds_proba)
            self.precisions_intra.append(precision)
            self.recalls_intra.append(recall)
            print(pipeline_name,":")    
            print("Score : ", score)
            print("Precision : ", precision)
            print("Recalll : ", recall)
            print("Confusion matrix : ", confusion_matrix)
            print("_________________")


    def plot_classification_by_electrodes(self, indices, lists, labels):

        for lst in range(len(lists)):
            plt.plot(indices, lists[lst], label=labels[lst])
        #plt.plot(x, y_f1, label='F1 score subjects')
        plt.legend()
        plt.title("Classification scores by electrodes number")
        plt.xlabel("Number of selected electrodes")
        plt.ylabel("Classification score")
        plt.show()        




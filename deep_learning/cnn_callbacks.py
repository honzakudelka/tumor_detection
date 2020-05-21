"""
This file provides an implementation of specific training callbacks which enable interaction
and monitoring of the learning progress.

Author:
    Jan Hering (BIA/CMP)
    hering.jan@fel.cvut.cz

"""
import numpy as np
import tflearn
from tflearn.callbacks import Callback
from tflearn.utils import feed_dict_builder
from sklearn.metrics import roc_auc_score


class PredictionCallback(Callback):
    """
    Class PredictionCallback implements computes probability output
    of the (current state) network on the validation set.

    :param predictor     the predictor (can be obtained from the DNN model)
    :param val_feed_dict feed dictionary constructed
    :param validation_step the callback will be active at each validation_step
    :param session      tf.session associated with the training phase
    :param labels       validation class labels
    """
    def __init__(self, predictor, t_inputs, data, labels, validation_step, session,
                 batch_size):

        super(Callback, self).__init__()
        # parameters
        self.validation_step = validation_step
        self.predictor = predictor
        self.t_inputs = t_inputs
        self.data = data
        self.val_y = [1 if l > 0 else -1 for l in labels]
        self.total_n = len(self.data)
        self.tf_session = session
        self.batch_size = batch_size

        # control variable
        self.last_run_step = 0

    def on_batch_end(self, training_state, snapshot=False):

        if training_state.step // self.validation_step > self.last_run_step:
            self.last_run_step += 1
            tflearn.is_training(False, session=self.tf_session)

            out_proba = np.array([])
            n_rep = self.total_n // self.batch_size + 1
            for rep in range(n_rep):

                fdict = feed_dict_builder(self.data[rep*self.batch_size:(rep+1)*self.batch_size],
                                          None, self.t_inputs, None)
                pproba = self.predictor.predict(fdict)

                out_proba = np.concatenate([out_proba, 1-pproba[:, 0]])

            sc = roc_auc_score(self.val_y, out_proba)
            print("Validation AUC: "+str(sc))

        else:
            pass


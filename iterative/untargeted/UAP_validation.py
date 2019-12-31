# Main function for UAP validation
# iterative greedy algorithm is used for targeted UAP generation.
# inputs to the "validation" function are UAP vector and the target model and the test set.
# outputs of "validation" function are SNR and Attack Success Rate (ASR) on test set.

import numpy as np
from misc import energy_np

def validation(v, model_end, X_test):

    num_false_after_perturbation=0
    snr_t=0

    for s in range(X_test.shape[0]):

        wav_test_pert=np.expand_dims(X_test[s], axis=0)+v
        wav_test_pert = np.clip(wav_test_pert, 0, 1)

        snr = 20 * np.log10(energy_np(np.squeeze(X_test[s]))/energy_np(wav_test_pert-X_test[s]))
        snr_t=snr+snr_t

        if np.argmax(model_end.predict(np.expand_dims(X_test[s], axis=0)))!=np.argmax(model_end.predict(wav_test_pert)):
            num_false_after_perturbation=num_false_after_perturbation+1

    return snr_t / float(X_test.shape[0]), float(num_false_after_perturbation)/float(X_test.shape[0])

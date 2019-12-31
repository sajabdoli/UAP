# Main function for UAP validation
# Penalty-based algorithm is used for untargeted UAP generation.
# inputs to the "validation" function are:
    # v_trainset: UAP vector
    # model_end: end to end audio classifier used as target model
    # X_test: Test set of audio samples
    
# outputs of "validation" function are pertubation SNR and Attack Success Rate (ASR) on test set.

import numpy as np
from misc import energy_np

def validation(v_trainset, model_end, X_test):
    
    boxmin=0 #Minimum signal value (default 0).
    boxmax=1 #Maximum signal value (default 1).
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.

    num_false_after_perturbation=0
    snr_t=0
    
    for s in range(X_test.shape[0]):
        
        X_test_arced=np.arctanh((X_test[s] - boxplus) / boxmul * 0.999999)
        audio_pert=np.add(X_test_arced,v_trainset)      
        audio_pert=np.tanh(audio_pert)* boxmul + boxplus

        test_s=np.squeeze(X_test_arced)
        test_s=np.tanh(test_s)*boxmul + boxplus

        snr = 20 * np.log10(energy_np(np.squeeze(test_s))/energy_np(np.squeeze(audio_pert)-test_s))
        snr_t=snr+snr_t

        classified = np.argmax(model_end.predict([np.expand_dims(audio_pert, axis=0)]), axis=1)

        correct=np.argmax(model_end.predict(np.expand_dims(X_test[s], axis=0)))

        if (classified != correct):
            num_false_after_perturbation = num_false_after_perturbation + 1

    fooling_rate_test = float(num_false_after_perturbation)/float(X_test.shape[0])
    SNR_avg = snr_t / float(X_test.shape[0])
    
    return SNR_avg, fooling_rate_test
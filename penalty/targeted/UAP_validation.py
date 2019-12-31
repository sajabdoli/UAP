# Main function for UAP validation
# Penalty-based algorithm is used for targeted UAP generation.
# inputs to the "validation" function are:
    # v_trainset: UAP vector
    # Target_class: index to target class
    # model_end: end to end audio classifier used as target model
    # X_test: Test set of audio samples
    
# outputs of "validation" function are pertubation SNR and Attack Success Rate (ASR) on test set.

#target_names=['air_conditioner'=0,'car_horn'=1,'children_playing'=2,'dog_bark'=3,
#              'drilling'=4,'engine_idling'=5,'gun_shot'=6,'jackhammer'=7,'siren'=8,'street_music'=9]

import numpy as np
from misc import energy_np

def validation(v_trainset, Target_class, model_end, X_test):
    
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

        snr = 20.0 * np.log10(energy_np(test_s)/energy_np(np.squeeze(audio_pert)-test_s))
        snr_t=snr+snr_t

        classified = np.argmax(model_end.predict([np.expand_dims(audio_pert, axis=0)]), axis=1)
        
        if (classified == Target_class):
            num_false_after_perturbation = num_false_after_perturbation + 1

    #print ('num_false_after_perturbation', num_false_after_perturbation)
    fooling_rate_test =float(num_false_after_perturbation)/float(X_test.shape[0])
    SNR_avg = snr_t / float(X_test.shape[0])
    
    return SNR_avg, fooling_rate_test
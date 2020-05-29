# Main function for UAP validation
# iterative greedy algorithm is used for targeted UAP generation.
# inputs to the "validation" function are UAP vector and target class number, the target model and the test set.
# outputs of "validation" function are pertubation SNR and Attack Success Rate (ASR) on test set.
#target_names=['air_conditioner'=0,'car_horn'=1,'children_playing'=2,'dog_bark'=3,
#              'drilling'=4,'engine_idling'=5,'gun_shot'=6,'jackhammer'=7,'siren'=8,'street_music'=9]

import numpy as np
from misc import energy_np

def validation(v, Target_class, model_end, X_test):

    num_false_after_perturbation=0
    snr_t=0

    for s in range(X_test.shape[0]):

        v_fin=np.expand_dims(X_test[s], axis=0)+v
        
        snr = 20.0 * np.log10(energy_np(np.squeeze(X_test[s]))/energy_np(v_fin-X_test[s]))
        
        snr_t=snr+snr_t

        classified = np.argmax(model_end.predict([v_fin]))

        if (classified == Target_class):
            num_false_after_perturbation = num_false_after_perturbation + 1

    return snr_t / float(X_test.shape[0]), float(num_false_after_perturbation)/float(X_test.shape[0])

# Main function for UAP generation
# iterative greedy algorithm is used for targeted UAP generation.
# inputs to the "un_pert_gen" function are:
 #the class number of the target class, 
 #the target model, batch of audio samples, 
 #DDN attacker instance and tf session.
 #param xi: controls the l_p magnitude of the perturbation (default = 0.12)
# outputs of "un_pert_gen" function are pertubation vector=v and Attack Success Rate (ASR) on train set.
 
#target_names=['air_conditioner'=0,'car_horn'=1,'children_playing'=2,'dog_bark'=3,
#              'drilling'=4,'engine_idling'=5,'gun_shot'=6,'jackhammer'=7,'siren'=8,'street_music'=9]

from misc import proj_lp
import numpy as np


def un_pert_gen(Target_class,model_end, X, attacker, sess, xi=0.12,delta=0.1):
    

    max_iter_uni = 100  # termination criterion
    p = np.inf

    v = 0.0
    fooling_rate = 0.0
    num_wav =  np.shape(X)[0] # The audio samples should be stacked ALONG FIRST DIMENSION
    itr =0

    while fooling_rate < 1-delta and itr < max_iter_uni:

        print ('Starting pass number ', itr)
        # Go through the data set and compute the perturbation increments sequentially

        for k in range(0, num_wav):
            cur_wav = X[k:(k+1), :, :]
            cur_wav_pert = cur_wav+v

            cur_wav_pert = np.clip(cur_wav_pert, 0, 1)

            if int(np.argmax(np.array(model_end.predict(cur_wav_pert)))) != Target_class:
                    print('>> k = ', k, ', pass #', itr)

                    # Compute adversarial perturbation using DDN

                    adv = attacker.attack(sess, cur_wav_pert, [Target_class])
                    
                    dr = (adv - cur_wav_pert)

                    v = v + dr

                    # Project on lp ball

                    v = proj_lp(v, xi, p)

        itr = itr + 1
        print ('pass #', itr)

        # Perturb the dataset with computed perturbation

        X_perturbed = X + v

        labels_target = np.zeros((num_wav))
        est_labels_pert = np.zeros((num_wav))

        num_batches = np.int(np.ceil(np.float(num_wav) / np.float(num_wav)))

        # Compute the estimated labels in batches    
        for ii in range(0, num_batches):

            m = (ii * num_wav)
            M = min((ii+1)*num_wav, num_wav)

            labels_target[m:M] = Target_class
            est_labels_pert[m:M] = np.argmax(model_end.predict(X_perturbed[m:M, :, :]), axis=1).flatten()

            # Compute the fooling rate

            fooling_rate = float(np.sum(est_labels_pert == labels_target) / float(num_wav))
            print('FOOLING RATE = ', fooling_rate)
            
    return v, fooling_rate




# Main function for UAP generation
# iterative greedy algorithm is used for untargeted UAP generation.
# inputs to the "un_pert_gen" function are:
 #the target model, batch of audio samples (X) and their true label (Y), 
 #DDN attacker instance, tf session.
 #param xi: controls the l_p magnitude of the perturbation (default = 0.2)
# outputs of "un_pert_gen" function are pertubation vector=v and Attack Success Rate (ASR) on train set.
 

from misc import proj_lp
import numpy as np


def un_pert_gen(model_end, X, Y, attacker, sess, xi=0.2, delta=0.1):
    
   
    max_iter_uni = 100  # termination criterion
    
    p = np.inf

    v = 0.0
    fooling_rate = 0.0
    num_wav =  np.shape(X)[0] # The images should be stacked ALONG FIRST DIMENSION
    itr =0

    while fooling_rate < 1-delta and itr < max_iter_uni:

        print ('Starting pass number ', itr)
        # Go through the data set and compute the perturbation increments sequentially

        for k in range(0, num_wav):
            cur_wav = X[k:(k+1), :, :]
            cur_wav_pert=cur_wav+v

            cur_wav_pert = np.clip(cur_wav_pert, 0, 1)

            if int(np.argmax(np.array(model_end.predict(cur_wav))))==int(np.argmax(np.array(model_end.predict(cur_wav_pert)))):
                    print('>> k = ', k, ', pass #', itr)

                    # Compute adversarial perturbation using DDN

                    adv = attacker.attack(sess, cur_wav_pert, [Y[k]])
                    #print ('True:', [Y[k]])
                    #print ('Classified_DDN:',np.argmax(np.array(model_end.predict(adv))) )
                    dr = (adv - cur_wav_pert)

                    v = v + dr

                    v = proj_lp(v, xi, p)

        itr = itr + 1
        print ('pass #', itr)

        # Perturb the dataset with computed perturbation\

        X_perturbed = X + v

        est_labels_orig = np.zeros((num_wav))
        est_labels_pert = np.zeros((num_wav))

        num_batches = np.int(np.ceil(np.float(num_wav) / np.float(num_wav)))

        # Compute the estimated labels in batches    
        for ii in range(0, num_batches):

            m = (ii * num_wav)
            M = min((ii+1)*num_wav, num_wav)
            est_labels_orig[m:M] = np.argmax(model_end.predict(X[m:M, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(model_end.predict(X_perturbed[m:M, :, :]), axis=1).flatten()

            # Compute the fooling rate

            fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_wav))
            print('FOOLING RATE = ', fooling_rate)
            
    return v, fooling_rate




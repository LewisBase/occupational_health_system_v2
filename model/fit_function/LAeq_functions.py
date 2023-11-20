import numpy as np

from constants.auditory_constants import AuditoryConstants


class LAeqFunction():
    
    def LAeq_single(x, b0, b1):
        return b0 + b1 * x

        
    def lambda_total(x, b0, b1, l):
        LAeq, kurtosis_mean = x
        return b0 + b1 * (LAeq + l * (np.log10(kurtosis_mean /
                                               AuditoryConstants.BASELINE_NOISE_KURTOSIS)))


    def lambda_total_duration(x, b0, b1, l, b2):
        LAeq, kurtosis_mean, duration = x
        return b0 + b1 * (LAeq + l * (np.log10(kurtosis_mean /
                                               AuditoryConstants.BASELINE_NOISE_KURTOSIS))) + b2 * duration
    
    def lambda_fix_duration(x, b2):
        x_feature, fix_params = x_feature
        LAeq, kurtosis_mean, duration = x
        b0, b1, l = fix_params
        return b0 + b1 * (LAeq + l * (np.log10(kurtosis_mean /
                                               AuditoryConstants.BASELINE_NOISE_KURTOSIS))) + b2 * duration

    def lambda_segment_ari(x, b0, b1, l):
        SPL_dBA, kurtosis = x
        return b0 + b1 * 10 * np.log10(np.mean(10**(SPL_dBA/10)*(kurtosis /
                                                         AuditoryConstants.BASELINE_NOISE_KURTOSIS)**(l/10)))


    def lambda_segment_ari_duration(x, b0, b1, l, b2):
        SPL_dBA, kurtosis, duration = x
        return b0 + b1 * 10 * np.log10(np.mean(10**(SPL_dBA/10)*(kurtosis /
                                                         AuditoryConstants.BASELINE_NOISE_KURTOSIS)**(l/10))) + b2 * duration

    def lambda_segment_geo(x, b0, b1, l):
        SPL_dBA, kurtosis = x
        return b0 + b1 * np.mean(SPL_dBA + l * np.log10(kurtosis /
                                                        AuditoryConstants.BASELINE_NOISE_KURTOSIS))

    def LAeq_logistic(x, a, b, c):
        return a / (1 + np.exp((b - x) / c)) 

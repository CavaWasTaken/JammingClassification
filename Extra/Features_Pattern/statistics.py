# ho un csv, voglio fare delle statistiche su di esso per colonne. Le statistiche sono : features_list = [mean_, median_, std_, mad_, rms_, percentile_25th_, percentile_75th_,
#                   iqr_, skewness_, kurtosis_, entropy_, max_power_win_, freq_max_power_,
# #                  mean_power_win_, pentropy_mean_, pentropy_std_]

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch

# Carica i CSV globalmente
df1 = pd.read_csv('./extracted_features_No_Antenna.csv')
df2 = pd.read_csv('./extracted_features_With_Antenna.csv')

def calculate_statistics(df):
    # Calcola media e deviazione standard per ogni colonna partendo dalla riga 1
    means = df.iloc[1:].mean(numeric_only=True).values
    stds = df.iloc[1:].std(numeric_only=True).values

    print('Media per colonna:')
    print(means)
    print('\nDeviazione standard per colonna:')
    print(stds)
    
    return means, stds

# Calcola statistiche per entrambi i dataset
means1, stds1 = calculate_statistics(df1)

print ('\n-----------------------------------\n')

# Calcola statistiche con antenna
means2, stds2 = calculate_statistics(df2)

print ('\n-----------------------------------\n')


# calcolo la differenza tra le due statistiche senza modulo
difference_means = means1 - means2
difference_stds = stds1 - stds2
print('\nDifferenza tra le medie per colonna:')
print(difference_means)
print('\nDifferenza tra le deviazioni standard per colonna:')
print(difference_stds)

# calcolo altre informazioni per evidentiare differenze tra le due statistiche
# differenza percentuale tra le medie
percentage_difference_means = (difference_means / means1) * 100
print('\nDifferenza percentuale tra le medie per colonna:')
print(percentage_difference_means)

# calcolo percentili 25 e 75 per entrambe le statistiche
percentile_25th_means1 = df1.iloc[1:].quantile(0.25, numeric_only=True).values
percentile_75th_means1 = df1.iloc[1:].quantile(0.75, numeric_only=True).values
percentile_25th_means2 = df2.iloc[1:].quantile(0.25, numeric_only=True).values
percentile_75th_means2 = df2.iloc[1:].quantile(0.75, numeric_only=True).values
print('\nPercentile 25th per colonna (No Antenna):')
print(percentile_25th_means1)
print('\nPercentile 75th per colonna (No Antenna):')
print(percentile_75th_means1)
print('\nPercentile 25th per colonna (With Antenna):')
print(percentile_25th_means2)
print('\nPercentile 75th per colonna (With Antenna):')
print(percentile_75th_means2)

# calcolo differenza percentuale tra i percentili 25 e 75
percentage_difference_percentile_25th = ((percentile_25th_means1 - percentile_25th_means2) / percentile_25th_means1) * 100
percentage_difference_percentile_75th = ((percentile_75th_means1 - percentile_75th_means2) / percentile_75th_means1) * 100
print('\nDifferenza percentuale tra i percentili 25th per colonna:')
print(percentage_difference_percentile_25th)
print('\nDifferenza percentuale tra i percentili 75th per colonna:')
print(percentage_difference_percentile_75th)



# # calcolo differenza delle due statistiche
# def calculate_difference(mean1, std1, mean2, std2):
#     difference = {}
#     for key in mean1.index:
#         difference[key] = {}
#         for column in mean1[key]:
#             difference[key][column] = mean1[key][column] - mean2[key][column]
#     for key in std1.index:
#         difference[key] = {}
#         for column in std1[key]:
#             difference[key][column] = std1[key][column] - std2[key][column]
#     return difference
# # Calcola la differenza tra le due statistiche
# difference = calculate_difference((means1, stds1), (means2, stds2))
# print('Differenza tra le statistiche:')
# print(difference)





####################
# Multi-scale Sample Entropy
####################

#Original Entropy Functions
import numpy as np
# Coarse graining array into multi-scale vectors
def coarse_graining(array: np.array, m_ent_type = 'Original', delay = 2):
    N = len(array)
    # Raises exceptions if some conditions are not statisifed.
    if not (m_ent_type == 'Original' or m_ent_type == 'Modified'):
        raise Exception("Please type in put Modified or Original as your m_ent_type")
    
    if not isinstance(delay,int):
        raise Exception("Please use an integer for tau")
        
    if delay <= 1 :
        raise Exception("Please put an delay that is more than one")
        
    if delay > N :
        raise Warning("delay is more than the length of the array, returning None")
        return None
    
    if  (np.max(array) == np.min(array)):
        if m_ent_type == 'Original':
            return array[range(0,floor(N/delay))]
        return array[range(0,1+N-delay)]
    
    if (m_ent_type =='Original'):

        coarsed_array = []
        coarsed_array.append(array[0:N])

        for t in range(2, delay+1):
            test = []
            for i in range(0, N, t):
                test.append(np.average(array[i:(i+t)]))
            coarsed_array.append(test)     

    if (m_ent_type =='Modified'):
        coarsed_array = list(array[range(0,1+N-delay)])
        for t in range(2,delay+1):
            array_to_add = list(array[range(t-1,t+N-delay)])
            coarsed_array = coarsed_array + array_to_add
    return coarsed_array

#################### Multiscale Entropy for univariate case

def multi_scale_entropy(array: np.array, m_ent_type = "Original", order = 3, delay = 3, metric = "chebyshev"):
    coarsed_array = coarse_graining(array, m_ent_type = m_ent_type, delay = delay)
    len(coarsed_array)
    entropylist = list()
    for item in coarsed_array:   
        entropylist.append(sample_entropy(item, order=order, metric=metric))
        
    finalen = np.average(entropylist)
    return finalen


##################Nomralizing data

def normalize(list_in: list):
    normed_data = list()
    value = list_in.values
    value = value.reshape((len(value), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(value)
    normalized = scaler.transform(value)
    normalized = np.array(normalized)
    normed_data.append(normalized)
    normed_data = np.concatenate(normed_data)
    return np.concatenate(normed_data)


###################Multivariate Case
def MMSE(array:np.array, m_ent_type = "Original", order = 3, delay = 3, metric = "chebyshev"):
    coarsed_array = list()
    for item1 in array:
        c_a = coarse_graining(item1, m_ent_type = m_ent_type, delay = delay)
        coarsed_array += c_a

    entropylist = list()
    for item in coarsed_array:
        entropylist.append(sample_entropy(item, order=order, metric=metric))
    finalen = np.average(entropylist)
    return finalen

################################################################################################################################################



########################
#Tensorflow Translation#
########################


import tensorflow as tf


def fir_filter(signal, window, stride=1):
    #Simple 1-D FIR filter to smooth signal, will average on window size
    size = tf.shape(signal)[0]
    tmp = tf.nn.avg_pool1d(tf.reshape(
        signal, [1, size, 1]), window, stride, padding='SAME')
    return tf.reshape(tmp, [tf.size(tmp)])

@tf.function
def sample_entropy(window, r):
    #set a time lag of 1
    xm2 = tf.stack([window[:-1], window[1:]], axis=1)
    size2 = tf.shape(xm2)[0] #size of first dimension
    xm2e = tf.tile(tf.expand_dims(xm2, 1), [1, size2, 1]) #creates time lags ex: [1,2,3,...], [2,3,4,..], etc.
    xm2eT = tf.transpose(xm2e, perm=[1, 0, 2])
    B = tf.shape(tf.where(tf.less_equal(tf.reduce_max(tf.abs(xm2e - xm2eT), axis=2), r)))[0] - size2 #count probability of repeat pattern in lags
    xm3 = tf.stack([window[:-2], window[1:-1], window[2:]], axis=1)
    size3 = tf.shape(xm3)[0]
    xm3e = tf.tile(tf.expand_dims(xm3, 1), [1, size3, 1])
    xm3eT = tf.transpose(xm3e, perm=[1, 0, 2])
    A = tf.shape(tf.where(tf.less_equal(tf.reduce_max(tf.abs(xm3e - xm3eT), axis=2), r)))[0] - size3
    return tf.cast(-tf.math.log(A/B), tf.float32)


def multiscale_entropy(window, r):
    """ Using only 3 scales now to reduce execution time
        Also limiting time window to 4 seconds/4000 ms """
    #r = 0.3 *std(timeseries) is a common estimation
    window = window[:4000]
    entropies = tf.stack([
        sample_entropy(fir_filter(window, 8, 8), r),
        sample_entropy(fir_filter(window, 16, 16), r),
        sample_entropy(fir_filter(window, 32, 32), r)
    ])
    return tf.cast(tf.reduce_mean(entropies, axis=0), tf.float32)
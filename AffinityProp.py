# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:18:02 2019
 
@author: MWorthington
"""
 
import numpy as np
import sklearn.cluster
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
#import distance
 
def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
      
    res = min([LD(s[:-1], t)+1,
               LD(s, t[:-1])+1,
               LD(s[:-1], t[:-1]) + cost])
    return res
 
stop_words = stopwords.words('english')
words = "As stated in my initial appeal on arrival at the hotel to check-in the Reception instructed we leave our registration number for entry in their system that parking is complementary. When we initially arrived at about 16:00 hours, there were no parking space so I had to disembarked and asked for the car to be parked outside the hotel with intention to bring the car back into the Hotel Parking lot when spaces are available; that was after the car registration number was left with the receptionist that the car is a Mobility car and need Blue Badge parking or any space to park to facilitate easy access. The car was moved in later in the evening when spaces became available. I accept that the signage around the Hotel were visible but on the advise and instruction of the Receptionist no payment was made on the understanding that the parking was complementary and that my registration number was noted on their system. I have tried unsuccessfully to get a collaborating statement from the Hotel management on the event of that evening. As a law abiding citizen, it is my appeal that there was no deliberate intention on my part to contravene any parking contract on the grounds of the hotel if not for the assurance I got from the Receptionist on the day in question that parking was complimentary for checking-in guest. A room was booked for me by the Celebrant of the event attended due to special need of the blue badge holder..".split(" ") #Replace this line
words = np.asarray(words)
words = [word for word in words if word not in stop_words]
words = np.asarray(words)
lev_similarity = -1*np.array([[LD(w1,w2) for w1 in words] for w2 in words])
 
affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))
   
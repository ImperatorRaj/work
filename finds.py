def findS(data, positive_label = '1'):
    examples = data[:, :-1]
    labels = data[:, -1]

    n_features = examples.shape[1]
    
    hypothesis = ['0'] * n_features
    
    for example, label in zip(examples, labels):
        if label != positive_label:
            continue
        
        for ind, attribute in enumerate(example):
            if hypothesis[ind] == '0':
                hypothesis[ind] = attribute
            elif hypothesis[ind] != attribute:
                hypothesis[ind] = '?'
                
    return hypothesis

import numpy as np
import pandas as pd
data = np.array([['Morning', 'Sunny', 'Warm', 'Yes', 'Mild', 'Strong', 'Yes'],
['Evening', 'Rainy', 'Cold', 'No', 'Mild', 'Normal', 'No'],
['Morning', 'Sunny', 'Moderate', 'Yes', 'Normal', 'Normal', 'Yes'],
['Evening', 'Sunny', 'Cold', 'Yes', 'High', 'Strong', 'Yes']])
print(findS(data, positive_label = 'Yes'))

data1=np.array([['Big', 'Red', 'Circle','No'],
       ['Small', 'Red', 'Triangle','No'],
       ['Small', 'Red', 'Circle','Yes'],
       ['Big', 'Blue', 'Circle','No'],
       ['Small', 'Blue', 'Circle','Yes']])
print(findS(data1,positive_label='Yes'))

import numpy as np
import pandas as pd
X = [1,2,3,4,5]
y = [1.2,1.8,2.6,3.2,3.8]
X = np.array(X)
y = np.array(y)
import warnings
warnings.filterwarnings('ignore')
df = pd.DataFrame({'Week': X,'Sales': y})
df.to_csv('sales_per_week.csv',index=False)
sdf = pd.read_csv('sales_per_week.csv')
sdf.head()
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split

lr = LinearRegression()
sX = sdf[['Week']]
sy = sdf['Sales']
lr.fit(sX,sy)
print(lr.predict([[7]])[0])
print(lr.predict([[12]])[0])

hX = np.array([29,15,33,28,39])
hy = np.array([0,0,1,1,1])
logr = LogisticRegression(random_state=0)
hdf = pd.DataFrame({'Hours': hX, 'Result': hy})
hdf.to_csv('hours_result.csv',index=False)

datf = pd.read_csv('hours_result.csv')
X_train = datf[['Hours']]
y_train = datf['Result']
logr.fit(X_train,y_train)
logr.score(X_train,y_train)

logr.predict([[33]])[0]
from scipy.optimize import fsolve

# Define the logistic function
def logistic_function(hours):
    log_odds = -64 + 2 * hours
    return 1 / (1 + np.exp(-log_odds))

# Function to find the hours for passing probability greater than 95%
def find_hours(prob_threshold):
    def equation(hours):
        return logistic_function(hours) - prob_threshold
    # Initial guess for hours
    initial_guess = 0
    # Solve the equation
    hours_solution = fsolve(equation, initial_guess)
    return hours_solution[0]

passing_prob_threshold = 0.95
hours_needed = find_hours(passing_prob_threshold)
print(hours_needed)

import pandas as pd
import scipy
import numpy as np
import statistics
from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Import Data

data = pd.read_excel('/Users/architjavkhedkar/Downloads/assignment/data.xlsx')

data

#Filter out the date
data_filt = data.drop('DATE', axis = 1)

data_filt

#Training Data 189 data points
data_train = data_filt.iloc[:189]

#Data for question 1, RPI
data_q1 = data_train['RPI']

data_q1

#Selecting y2 as values
y2 = data_q1.values

y2

"""# Q1.a"""

#Initialising Parameters
T = 189
p1 = 0.8
mu1 = y2.mean()
mu2 = y2.mean()
sigma1 = 0.5 * y2.std()
sigma2 = 1.5 * y2.std()

#Storing initialising Parameters
parameter_vectors1 = [0.8,y2.mean(),y2.mean(),0.5 * y2.std(),1.5 * y2.std()]

# Log Likelihood Function for Finite Mixture Model
def log_likelihood(parameters, y2):
    p1 = parameters[0]
    p2 = 1-p1
    mu1 = parameters[1]
    mu2 = parameters[2]
    sigma1 = parameters[3]
    sigma2 = parameters[4]


    #likelihood formula f(yt;theta) combined for both regimes

    likelihood = p1 * norm.pdf(y2, mu1, sigma1) + (p2) * norm.pdf(y2, mu2, sigma2)

    #log likelihood per observation
    log_likelihood = np.log(likelihood)

    #sum over all observations
    ll_value = np.sum(log_likelihood)

    #Return negative log likelihood
    return -ll_value

#Optimisation Model
def optimise_and_print(parameter_vector, ll_func):
    res = minimize(ll_func, parameter_vector, args=(y2,), method = 'L-BFGS-B',
                bounds =[(0,1), (-100, 100), (-100,100), (0,100), (0,100)], tol = 1e-6) # set bounds for probability between 0,1 and for standard deviation such that it can't be negative


    # Estimated values
    estimated_parameters1 = res.x

    p1_est = estimated_parameters1[0]
    print ('p1_est =', p1_est)
    p2_est = 1 - p1_est
    print('p2_est =', p2_est)
    mu1_est = estimated_parameters1[1]
    print ('mu1_est =', mu1_est)
    mu2_est = estimated_parameters1[2]
    print ('mu2_est =', mu2_est)
    sigma1_est = estimated_parameters1[3]
    print ('sigma1_est =', sigma1_est)
    sigma2_est = estimated_parameters1[4]
    print ('sigma2_est =', sigma2_est)

    # Log Likelihood values
    optimised_log_likelihood = -res.fun
    print('Log Likelihood values', optimised_log_likelihood)
    return  p1_est, p2_est, mu1_est, mu2_est, sigma1_est, sigma2_est

print('Part A Estimates and Log Likelihood Values')
optimise_and_print(parameter_vectors1, log_likelihood)

# Probability of being in state 2 if value for the series y2,t equal to 0
y_obs = 0
p1_est, p2_est, mu1_est, mu2_est, sigma1_est, sigma2_est = optimise_and_print(parameter_vectors1, log_likelihood)
#Calculating Property
probability_recession = p2_est*norm.pdf(y_obs, mu2_est, sigma2_est)/(p2_est*norm.pdf(y_obs, mu2_est, sigma2_est) + p1_est*norm.pdf(y_obs, mu1_est, sigma1_est))
print('Probability of being in Recession if y2,t = 0 : ', probability_recession)

"""# Q1.b"""

#Given
p11 = 0.8
p22 = 0.8

#Initialising Vectors
initial_vectors = [0.8, 0.8, y2.mean(), y2.mean(), 0.5*y2.std(), 1.5*y2.std()]

#Functions for State 1 Initialisations

#Function to run hamilton filter
def hamilton_filter1(p11, p22, mu1,mu2,sigma1, sigma2, y2):
    #Transition Matrix
    P = [[p11 , 1- p22], [1-p11, p22 ]]

    #Length of data in y2
    T = len(y2)
    #Create 2 Dimensional Arrays on length T
    predicted_xi = np.zeros((2,T))
    filtered_xi = np.zeros((2,T))
    likelihood = np.zeros((2,T))

    #Initialise the filter
    predicted_xi[:, 0] = [1,0]

    #iterate over the length of y2, to predict and filter.
    for i in range(T):
        #calculate likelihood based on value from data and given parameters
        likelihood[:, i] = [norm.pdf(y2[i], mu1, sigma1), norm.pdf(y2[i], mu2, sigma2)]

        #Run the filter to calculate filtered values based on the prediction and likelihood.
        filtered_xi[:, i] = (predicted_xi[:,i]* likelihood[:,i])/np.dot(predicted_xi[:,i], likelihood[:,i])
        #This check makes sure we don't exceed size of Array
        if i < T-1:
            #Update next prediction using the transition matrix and current filtered value.
            predicted_xi[:, i+1] = np.dot(P, filtered_xi[:, i])
    #print (filtered_xi)
    return filtered_xi, predicted_xi

# This function uses predicted values from the Hamilton filter to calculate log likelihood.
def log_likelihood_hamilton1(parameters, y2):
    p11 = parameters[0]
    p22 = parameters[1]
    mu1 = parameters[2]
    mu2 = parameters[3]
    sigma1 = parameters[4]
    sigma2 = parameters[5]



    #Runs the hamilton filter and extracts predicted values
    _,predicted_xi = hamilton_filter1(p11,p22,mu1,mu2,sigma1,sigma2,y2)

    #calculates log likelihood using predicted values

    ll = np.log(predicted_xi[0,:]* norm.pdf(y2, mu1, sigma1) + predicted_xi[1,:]* norm.pdf(y2, mu2, sigma2))
    total_ll = -np.sum(ll)


    return total_ll

# Optimisation function that uses L-BFGS-B to Maximize Likelihood (minimize negative log likelihood)
def optimise_and_print_hamilton1(parameter_vector, ll_func):
    # set bounds for probability between 0,1 and for standard deviation such that it can't be negative
    res = scipy.optimize.minimize(ll_func, parameter_vector, args=(y2,), method = 'L-BFGS-B',
                bounds =[(0,1), (0,1), (-100, 100),(-100, 100), (0,100), (0,100)], tol = 1e-6)


    # Estimated values
    estimated_parameters1 = res.x

    p11_est = estimated_parameters1[0]
    print ('p11_est =', p11_est)
    p22_est = estimated_parameters1[1]
    print ('p22_est =', p22_est)

    mu1_est = estimated_parameters1[2]
    print ('mu1_est =', mu1_est)
    mu2_est = estimated_parameters1[3]
    print ('mu2_est =', mu2_est)
    sigma1_est = estimated_parameters1[4]
    print ('sigma1_est =', sigma1_est)
    sigma2_est = estimated_parameters1[5]
    print ('sigma2_est =', sigma2_est)

    #Calculate Long Term Means
    long_term_mean1 = (1-p22_est)/(2-p11_est-p22_est)
    print ('Long Term Mean 1 =', long_term_mean1)
    long_term_mean2 = (1-p11_est)/(2-p11_est-p22_est)
    print ('Long Term Mean 2 =', long_term_mean2)
    # Log Likelihood values
    optimised_log_likelihood = -res.fun
    print('Log Likelihood values', optimised_log_likelihood)


    return p11_est, p22_est, mu1_est, mu2_est, sigma1_est, sigma2_est, long_term_mean1, long_term_mean2, optimised_log_likelihood

#Functions for State 2 Initialisations

#Function to run hamilton filter
def hamilton_filter2(p11, p22, mu1,mu2,sigma1, sigma2, y2):
    #Transition Matrix
    P = [[p11 , 1- p22], [1-p11, p22 ]]

    #Length of data in y2
    T = len(y2)
    #Create 2 Dimensional Arrays on length T
    predicted_xi = np.zeros((2,T))
    filtered_xi = np.zeros((2,T))
    likelihood = np.zeros((2,T))

    #Initialise the filter

    predicted_xi[:, 0] = [0,1]

    #iterate over the length of y2, to predict and filter.
    for i in range(T):
        #calculate likelihood based on value from data and given parameters
        likelihood[:, i] = [norm.pdf(y2[i], mu1, sigma1), norm.pdf(y2[i], mu2, sigma2)]

        #Run the filter to calculate filtered values based on the prediction and likelihood.
        filtered_xi[:, i] = (predicted_xi[:,i]* likelihood[:,i])/np.dot(predicted_xi[:,i], likelihood[:,i])
        #This check makes sure we don't exceed size of Array
        if i < T-1:
            #Update next prediction using the transition matrix and current filtered value.
            predicted_xi[:, i+1] = np.dot(P, filtered_xi[:, i])
    #print (filtered_xi)
    return filtered_xi, predicted_xi

# This function uses predicted values from the Hamilton filter to calculate log likelihood.
def log_likelihood_hamilton2(parameters, y2):
    p11 = parameters[0]
    p22 = parameters[1]
    mu1 = parameters[2]
    mu2 = parameters[3]
    sigma1 = parameters[4]
    sigma2 = parameters[5]



    #Runs the hamilton filter and extracts predicted values
    _,predicted_xi = hamilton_filter2(p11,p22,mu1,mu2,sigma1,sigma2,y2)

    #calculates log likelihood using predicted values

    ll = np.log(predicted_xi[0,:]* norm.pdf(y2, mu1, sigma1) + predicted_xi[1,:]* norm.pdf(y2, mu2, sigma2))
    total_ll = -np.sum(ll)


    return total_ll

# Optimisation function that uses L-BFGS-B to Maximize Likelihood (minimize negative log likelihood)
def optimise_and_print_hamilton2(parameter_vector, ll_func):
    # set bounds for probability between 0,1 and for standard deviation such that it can't be negative
    res = scipy.optimize.minimize(ll_func, parameter_vector, args=(y2,), method = 'L-BFGS-B',
                bounds =[(0,1), (0,1), (-100, 100),(-100, 100), (0,100), (0,100)], tol = 1e-6)


    # Estimated values
    estimated_parameters1 = res.x

    p11_est = estimated_parameters1[0]
    print ('p11_est =', p11_est)
    p22_est = estimated_parameters1[1]
    print ('p22_est =', p22_est)

    mu1_est = estimated_parameters1[2]
    print ('mu1_est =', mu1_est)
    mu2_est = estimated_parameters1[3]
    print ('mu2_est =', mu2_est)
    sigma1_est = estimated_parameters1[4]
    print ('sigma1_est =', sigma1_est)
    sigma2_est = estimated_parameters1[5]
    print ('sigma2_est =', sigma2_est)

    #Calculate Long Term Means
    long_term_mean1 = (1-p22_est)/(2-p11_est-p22_est)
    print ('Long Term Mean 1 =', long_term_mean1)
    long_term_mean2 = (1-p11_est)/(2-p11_est-p22_est)
    print ('Long Term Mean 2 =', long_term_mean2)
    # Log Likelihood values
    optimised_log_likelihood = -res.fun
    print('Log Likelihood values', optimised_log_likelihood)


    return p11_est, p22_est, mu1_est, mu2_est, sigma1_est, sigma2_est, long_term_mean1, long_term_mean2, optimised_log_likelihood

#Functions to run Long-Run Initialisation

#Function to run hamilton filter
def hamilton_filter3(p11, p22, mu1,mu2,sigma1, sigma2, y2):
    #Transition Matrix
    P = [[p11 , 1- p22], [1-p11, p22 ]]

    #Length of data in y2
    T = len(y2)
    #Create 2 Dimensional Arrays on length T
    predicted_xi = np.zeros((2,T))
    filtered_xi = np.zeros((2,T))
    likelihood = np.zeros((2,T))

    #Initialise the filter
    predicted_xi[:, 0] = [(1-p22)/(2-p11-p22), (1-p11)/(2-p11-p22)]

    #iterate over the length of y2, to predict and filter.
    for i in range(T):
        #calculate likelihood based on value from data and given parameters
        likelihood[:, i] = [norm.pdf(y2[i], mu1, sigma1), norm.pdf(y2[i], mu2, sigma2)]

        #Run the filter to calculate filtered values based on the prediction and likelihood.
        filtered_xi[:, i] = (predicted_xi[:,i]* likelihood[:,i])/np.dot(predicted_xi[:,i], likelihood[:,i])
        #This check makes sure we don't exceed size of Array
        if i < T-1:
            #Update next prediction using the transition matrix and current filtered value.
            predicted_xi[:, i+1] = np.dot(P, filtered_xi[:, i])
    #print (filtered_xi)
    return filtered_xi, predicted_xi

# This function uses predicted values from the Hamilton filter to calculate log likelihood.
def log_likelihood_hamilton3(parameters, y2):
    p11 = parameters[0]
    p22 = parameters[1]
    mu1 = parameters[2]
    mu2 = parameters[3]
    sigma1 = parameters[4]
    sigma2 = parameters[5]



    #Runs the hamilton filter and extracts predicted values
    _,predicted_xi = hamilton_filter3(p11,p22,mu1,mu2,sigma1,sigma2,y2)

    #calculates log likelihood using predicted values

    ll = np.log(predicted_xi[0,:]* norm.pdf(y2, mu1, sigma1) + predicted_xi[1,:]* norm.pdf(y2, mu2, sigma2))
    total_ll = -np.sum(ll)


    return total_ll

# Optimisation function that uses L-BFGS-B to Maximize Likelihood (minimize negative log likelihood)
def optimise_and_print_hamilton3(parameter_vector, ll_func):
    # set bounds for probability between 0,1 and for standard deviation such that it can't be negative
    res = scipy.optimize.minimize(ll_func, parameter_vector, args=(y2,), method = 'L-BFGS-B',
                bounds =[(0,1), (0,1), (-100, 100),(-100, 100), (0,100), (0,100)], tol = 1e-6)


    # Estimated values
    estimated_parameters1 = res.x

    p11_est = estimated_parameters1[0]
    print ('p11_est =', p11_est)
    p22_est = estimated_parameters1[1]
    print ('p22_est =', p22_est)

    mu1_est = estimated_parameters1[2]
    print ('mu1_est =', mu1_est)
    mu2_est = estimated_parameters1[3]
    print ('mu2_est =', mu2_est)
    sigma1_est = estimated_parameters1[4]
    print ('sigma1_est =', sigma1_est)
    sigma2_est = estimated_parameters1[5]
    print ('sigma2_est =', sigma2_est)

    #Calculate Long Term Means
    long_term_mean1 = (1-p22_est)/(2-p11_est-p22_est)
    print ('Long Term Mean 1 =', long_term_mean1)
    long_term_mean2 = (1-p11_est)/(2-p11_est-p22_est)
    print ('Long Term Mean 2 =', long_term_mean2)
    # Log Likelihood values
    optimised_log_likelihood = -res.fun
    print('Log Likelihood values', optimised_log_likelihood)


    return p11_est, p22_est, mu1_est, mu2_est, sigma1_est, sigma2_est, long_term_mean1, long_term_mean2, optimised_log_likelihood

#Initial State 1
optimise_and_print_hamilton1(initial_vectors, log_likelihood_hamilton1)

# Initial State 2
optimise_and_print_hamilton2(initial_vectors, log_likelihood_hamilton2)

# Long Term mean initialisation
optimise_and_print_hamilton3(initial_vectors, log_likelihood_hamilton3)

"""# Q1 c."""

data_pred = data_filt.iloc[189:]

data_pred_q1 = data_pred['RPI']

y2_prediction_data = data_pred_q1.values

#Run after running State 1 initialisation
p11_est, p22_est, mu1_est, mu2_est, sigma1_est, sigma2_est, long_term_mean1, long_term_mean2,optimised_log_likelihood = optimise_and_print_hamilton1(initial_vectors, log_likelihood_hamilton1)

# Get filtered_xi using the Hamilton filter with optimized parameters
filtered_xi, predicted_xi = hamilton_filter1(p11_est, p22_est, mu1_est, mu2_est, sigma1_est, sigma2_est, y2)

# Extracting filtered value for Q4 2019. Used for Q1 2020 prediction
filtered_val = filtered_xi[:,-1]
filtered_val

# Read prediction data
#y_prediction_data = pd.read_csv('C:/Users/jobbe/OneDrive/Documents/Studiejaar 7 (2024-2025)/Time Series/Assignment/assignment (1)/assignment/Forecasting Data - Time Series.csv')

# Extract RPI values
#y2_prediction_data = y_prediction_data['RPI'].values

# Q1 2020 value for RPI
y2_q12020 = y2_prediction_data[0]
y2_q12020

# Transition Matrix after optimisation
new_P = np.array([[p11_est, 1-p22_est], [1-p11_est, p22_est]])
new_P



# Calculate Predicted Probability
predicted_q1 = np.dot(new_P, filtered_val)

#calculate likelihood for filtered probability
likelihood_q1 = [norm.pdf(y2_q12020, mu1_est, sigma1_est), norm.pdf(y2_q12020, mu2_est, sigma2_est)]

#Calculate filtered values based on the prediction and likelihood.
filtered_prob_q12020 = (predicted_q1* likelihood_q1)/np.dot(predicted_q1, likelihood_q1)
filtered_prob_q12020
print('Probability of being in Recession in Q1 2020 is', predicted_q1[1])
print('Filtered Probability of being in State 2 is', filtered_prob_q12020[1])

"""# Q. 1.d"""

def hamilton_filter_1d(p11, p22, mu1,mu2,sigma1, sigma2, xi0_in, y2):

    #Transition Matrix
    P = np.array([[p11, 1 - p22], [1 - p11, p22]])


    #Length of data in y2
    T = len(y2)
    #Create 2 Dimensional Arrays on length T
    predicted_xi = np.zeros((2,T))
    filtered_xi = np.zeros((2,T))
    likelihood = np.zeros((2,T))

    #Initialise the filter, changes depending on initialization

    predicted_xi[:, 0] = xi0_in


    #iterate over the length of y2, to predict and filter.
    for i in range(T):
        #calculate likelihood based on value from data and given parameters
        likelihood[:, i] = [
            max(norm.pdf(y2[i], mu1, sigma1), 1e-10),  # Prevent underflow
            max(norm.pdf(y2[i], mu2, sigma2), 1e-10)
        ]
        #likelihood[:,i] = [norm.pdf(y2[i], mu1, sigma1), norm.pdf(y2[i], mu2, sigma2)]


        #Run the filter to calculate filtered values based on the prediction and likelihood.
        denominator = np.dot(predicted_xi[:, i], likelihood[:, i])+1e-6
        filtered_xi[:, i] = (predicted_xi[:,i]* likelihood[:,i])/denominator
        filtered_xi[:, i] /= np.sum(filtered_xi[:, i])
        #This check makes sure we don't exceed size of Array
        if i < T-1:
            #Update next prediction using the transition matrix and current filtered value.
            predicted_xi[:, i+1] = np.dot(P, filtered_xi[:, i])
            predicted_xi[:, i+1] /= np.sum(predicted_xi[:, i+1])


    return filtered_xi, predicted_xi

#First a hamilton smoother needs to be made
def hamilton_smoother(p11, p22, mu1, mu2, sigma1, sigma2, xi0_in, y):
    T= len(y)
    #this needs to be three dimensional
    P = np.array([[p11, 1 - p22],
              [1 - p11, p22]])
    #Now the Hamilton Filter needs to be done,
    filtered_xi, predicted_xi = hamilton_filter_1d(p11, p22, mu1, mu2, sigma1, sigma2,xi0_in, y) #we have two mu's and two sigma's, so must also be the case for smoother
    smoothed_xi = np.zeros((2,T))
    smoothed_xi[:,T-1] = filtered_xi[:,T-1]

    #Here the smoother will be run

    for i in range(1,T):
        t = T - 1 - i

        smoothed_xi[:, t] = filtered_xi[:, t] * (P.T @ (smoothed_xi[:, t + 1] / (predicted_xi[:, t + 1])))
        smoothed_xi[:, t] /= np.sum(smoothed_xi[:, t])


    #Getting a smoothed estimator for xi0

    xi0_out = xi0_in * (P.T @ ( (smoothed_xi[:,0]) / (predicted_xi[:,0]+1e-10) ))
    xi0_out /= np.sum(xi0_out)
    #print(smoothed_xi[:,0])


    #getting the cross terms
    Pstar = np.empty((2, 2, T))

    Pstar[:, :, 0] = P * (( (np.outer(smoothed_xi[:,0] , xi0_in)))  / (np.dot(predicted_xi[:, 0], np.ones(2))))
    #Pstar[:, :, 0] = np.dot(P, ((np.outer(smoothed_xi[:,0] , xi0_in)) / np.dot(predicted_xi[:,0], [1,1])))

    # Compute Pstar
    for t in range(1, T):
        for i in range(2):
            for j in range(2):
                Pstar[i, j, t] = (P[i, j] * filtered_xi[j, t - 1] * smoothed_xi[i, t]) / (predicted_xi[i, t] + 1e-10)
        Pstar[:, :, t] /= np.sum(Pstar[:, :, t].sum(axis=0))  # Normalize across columns
    return smoothed_xi, xi0_out, Pstar

def EM_step(p11,p22,mu1,mu2,sigma1,sigma2, xi0_in, y):

    T = len(y)

    smoothed_xi, xi0_out, Pstar = hamilton_smoother(p11,p22, mu1, mu2, sigma1, sigma2, xi0_in, y)

    #Initializing the empty vectors in order to sum over them.
    p11star = np.zeros(T)
    p12star = np.zeros(T)
    p21star = np.zeros(T)
    p22star = np.zeros(T)

    p1star = np.zeros(T)
    p2star = np.zeros(T)

    #get Pstar values in vectors
    for t in range(T):
        p11star[t]= Pstar[0,0,t]
        p12star[t] = Pstar[0,1,t]
        p1star[t] =  Pstar[0,0,t] + Pstar[0,1,t]
        p21star[t] = Pstar[1,0,t]
        p22star[t] = Pstar[1,1,t]
        p2star[t]  = Pstar[1,0,t] + Pstar[1,1,t]
        result = p1star[t]+p2star[t]
        #print(result)

    #This is the maximization step (M)
    p11_out = np.sum(p11star) / (xi0_out[0] + np.sum(p1star[:-1]))
    p22_out = np.sum(p22star) / (xi0_out[1] + np.sum(p2star[:-1]))
    mu1_out = np.sum( p1star * y ) / np.sum( p1star )
    mu2_out = np.sum( p2star * y ) / np.sum( p2star )
    sigma1_out= np.sqrt( np.sum( p1star * ( y - mu1 )**2 ) / sum(p1star))
    sigma2_out= np.sqrt( np.sum( p2star * ( y - mu2 )**2 ) / sum(p2star))

    return p11_out, p22_out, mu1_out, mu2_out, sigma1_out, sigma2_out, xi0_out

#New function for for the log likelihood
# This function uses predicted values from the Hamilton filter to calculate log likelihood.
def log_likelihood_hamilton_1d(parameters, xi0, y2):
    p11 = parameters[0]
    p22 = parameters[1]
    mu1 = parameters[2]
    mu2 = parameters[3]
    sigma1 = parameters[4]
    sigma2 = parameters[5]



    #Runs the hamilton filter and extracts predicted values
    _,predicted_xi = hamilton_filter_1d(p11, p22, mu1,mu2,sigma1, sigma2, xi0, y2)

    #calculates log likelihood using predicted values

    ll = np.log((predicted_xi[0,:]* norm.pdf(y2, mu1, sigma1) + predicted_xi[1,:]* norm.pdf(y2, mu2, sigma2)))
    #print(predicted_xi)
    total_ll = np.sum(ll)


    return total_ll

#In this code the above two functions are going to be used in order to estimate p11, p22, mu1, mu2, sigma 1 and sigma 2

#Model parameters
T = len(y2)
p11 = 0.8
p22 = 0.8
mu1 = y2.mean()
mu2 = y2.mean()
sigma1 = 0.5 * y2.std()
sigma2 = 1.5 * y2.std()
P = np.array([[p11, 1 - p22],
              [1 - p11, p22]])

rho_1 = 0.5

#Initialise using rho1 values

xi0_in = [rho_1, 1 - rho_1]

#doing EM...

iteration = 0
iterationLimit = 1001

#empty matrix for the parameters is
parameters_EM = np.empty((iterationLimit, 8))

for i in range(iterationLimit):

    #EM step
    p11, p22, mu1, mu2, sigma1, sigma2, xi0_out = EM_step(p11, p22 ,mu1, mu2, sigma1, sigma2, xi0_in, y2)

    #updating the rho's and saving estimated parameters
    xi0_in = xi0_out
    parameters_EM[i,:] =  [p11,p22,mu1,mu2,sigma1,sigma2, xi0_out[0], xi0_out[1]]
    #check at iteration 10
    if iteration == 9:
        parameters_10 = parameters_EM[9,:]
        xi0_10 = xi0_out
    #check for iteration 1000
    if iteration == 999:
        parameters_1000 = parameters_EM[999,:]
        xi0_1000 = xi0_out
    iteration = iteration + 1
print("The estimated parameters at the 10th iteration:", parameters_10)
print("The log likelihood of the 10th iteration", log_likelihood_hamilton_1d(parameters_10, xi0_10, y2))
print("The estimated parameters at the 1000th iteration:", parameters_1000)
print("The log likelihood of the 1000th iteration", log_likelihood_hamilton_1d(parameters_1000, xi0_1000, y2))

"""# Question 2"""

MainData_demeaned = data_train

#Processing the Data: De-meaning, creating lagged data, removing NaNs

# De-meaning the data
MainData_demeaned = data_train.apply(lambda x: x - x.mean())


# Create a lagged version of demeaned data
Lagged_MainData = MainData_demeaned.shift(1)

# Removing the first row to align both series and romoving the NaN in the lagged series. Also the indices are reset to ensure alignment in OLS later.
MainData_demeaned_aligned = MainData_demeaned.iloc[1:].reset_index(drop=True)
Lagged_MainData_aligned = Lagged_MainData.dropna().reset_index(drop=True)

"""# Q2 a"""

#Estimating the AR(1) models for the three series individually

# ---- For INDPRO ----
column1_data_aligned = MainData_demeaned_aligned[MainData_demeaned_aligned.columns[0]]
column1_lagged_aligned = Lagged_MainData_aligned[Lagged_MainData_aligned.columns[0]]

# Fit the AR(1) model for the first column
model1 = sm.OLS(column1_data_aligned, column1_lagged_aligned)
results1 = model1.fit()
phi1 = results1.params.iloc[0]
sigma_u_squared1 = results1.resid.var()
log_likelihood1 = -0.5 * len(results1.resid) * np.log(2 * np.pi * sigma_u_squared1) - (0.5 / sigma_u_squared1) * np.sum(results1.resid**2)

print(f"phi1 for INDPRO: {phi1}")
print(f"Residual variance for INDPRO: {sigma_u_squared1}")

# ---- For RPI ----
column2_data_aligned = MainData_demeaned_aligned[MainData_demeaned_aligned.columns[1]]
column2_lagged_aligned = Lagged_MainData_aligned[Lagged_MainData_aligned.columns[1]]

# Fit the AR(1) model for the second column
model2 = sm.OLS(column2_data_aligned, column2_lagged_aligned)
results2 = model2.fit()
phi2 = results2.params.iloc[0]
sigma_u_squared2 = results2.resid.var()

print(f"phi1 for RPI: {phi2}")
print(f"Residual variance for RPI: {sigma_u_squared2}")

# ---- For PAYEMS ----
# Extract the third column and create the lagged version
column3_data_aligned = MainData_demeaned_aligned[MainData_demeaned_aligned.columns[2]]
column3_lagged_aligned = Lagged_MainData_aligned[Lagged_MainData_aligned.columns[2]]

# Fit the AR(1) model for the third column
model3 = sm.OLS(column3_data_aligned, column3_lagged_aligned)
results3 = model3.fit()
phi3 = results3.params.iloc[0]
sigma_u_squared3 = results3.resid.var()

print(f"phi1 for for PAYEMS: {phi3}")
print(f"Residual variance for PAYEMS: {sigma_u_squared3}")

"""# Q2 b"""

#Log-Likelihood

def negative_log_likelihood_LL(parameter_vector, y):
    phi = parameter_vector[0] # persistence parameter
    Q = parameter_vector[1] # variance for the state equation
    R = parameter_vector[2] # variance for the observation equation

    #Giving Currrent parameters running kalman filter, initialisations are for the state is zero (as data demeaned) and variance sate Q/(1-phi^2)
    _, _, predicted_xi, predicted_P = KF_LL(phi, Q, R, y, 0, Q /(1 - phi**2))

    #Calculate the Log-likelihood
    LL = 0
    for t in range(len(y)):
        sigma = predicted_P[t] + R
        mu = predicted_xi[t]
        LL += -0.5*np.log(2*np.pi*sigma) - 0.5*((y[t] - mu)**2 / sigma)

    return -LL #negative as the main program is a minimisation problem (max LL ~ min -LL)

#Kalman Filter

def KF_LL(phi, Q, R, y, mu_init, P_init):
    T = len(y)
    predicted_xi = np.zeros(T) #the state predictions
    predicted_P = np.zeros(T) #the state variance predictions
    xi = np.zeros(T) #the state updates
    P = np.zeros(T) #the state variance updates

    #initialisation the first state predictions
    predicted_xi[0] = mu_init
    predicted_P[0] = P_init

    #calculating the first state update
    xi[0] = predicted_xi[0] + (predicted_P[0]/(predicted_P[0]+R))*(y[0] - predicted_xi[0])
    P[0] = predicted_P[0] - (predicted_P[0]/(predicted_P[0]+R)) * predicted_P[0]


    for t in range(1, T):
        #prediction step
        predicted_xi[t] = phi*xi[t-1]
        predicted_P[t] = phi**2 * P[t-1] + Q

        #update step
        xi[t] = predicted_xi[t] + (predicted_P[t] / (predicted_P[t] + R)) * (y[t] - predicted_xi[t])
        P[t] = predicted_P[t] - (predicted_P[t] / (predicted_P[t] + R)) * predicted_P[t]


    return xi, P, predicted_xi, predicted_P

#Series INDPRO
#Performing the Optimisation to obtain optimal parameters ML

#Select Data: 0 is y1, 1 is y2, 2 is y3
y = MainData_demeaned.iloc[:, 0]

#Setting initial values parameters and setting bounds for them, in order phi, sigma_theta^2, sigma_error^2
initial_guess = [0.5, 15, 25]
bounds = [(-1+1e-6 ,1-1e-6), (1e-6, None), (1e-6, None)] #bounds for theta, Q(sigma_theta^2)  and R (sigma_error^2)

options = {
'maxiter': 1000, #MaxIterations
'disp': True, #always for Matlab
}

#perform the optimisation
result = minimize(negative_log_likelihood_LL, initial_guess, args = (y,), bounds = bounds, options = options, method='SLSQP', tol=1e-8)

#Extracting the ML parameters that are optimal and the LogL value that was optained.
phi_ML, Q_ML, R_ML = result.x[:3]
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("ML_LogL:", -ML_LogL) #here we need -Negative log likelihood

#Series RPI
#Performing the Optimisation to obtain optimal parameters ML

#Select Data: 0 is y1, 1 is y2, 2 is y3
y = MainData_demeaned.iloc[:, 1]

#Setting initial values parameters and setting bounds for them, in order phi, sigma_theta^2, sigma_error^2
initial_guess = [0.5, 15, 25]
bounds = [(-1+1e-6 ,1-1e-6), (1e-6, None), (1e-6, None)] #bounds for theta, Q(sigma_theta^2)  and R (sigma_error^2)

options = {
'maxiter': 1000, #MaxIterations
'disp': True, #always for Matlab
}

#perform the optimisation
result = minimize(negative_log_likelihood_LL, initial_guess, args = (y,), bounds = bounds, options = options, method='SLSQP', tol=1e-8)

#Extracting the ML parameters that are optimal and the LogL value that was optained.
phi_ML, Q_ML, R_ML = result.x[:3]
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("ML_LogL:", -ML_LogL) #here we need -Negative log likelihood

#Series PAYEMS
#Performing the Optimisation to obtain optimal parameters ML

#Select Data: 0 is y1, 1 is y2, 2 is y3
y = MainData_demeaned.iloc[:, 2]

#Setting initial values parameters and setting bounds for them, in order phi, sigma_theta^2, sigma_error^2
initial_guess = [0.5, 15, 25]
bounds = [(-1+1e-6 ,1-1e-6), (1e-6, None), (1e-6, None)] #bounds for theta, Q(sigma_theta^2)  and R (sigma_error^2)

options = {
'maxiter': 1000, #MaxIterations
'disp': True, #always for Matlab
}

#perform the optimisation
result = minimize(negative_log_likelihood_LL, initial_guess, args = (y,), bounds = bounds, options = options, method='SLSQP', tol=1e-8)

#Extracting the ML parameters that are optimal and the LogL value that was optained.
phi_ML, Q_ML, R_ML = result.x[:3]
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("ML_LogL:", -ML_LogL) #here we need -Negative log likelihood

#Visulation of results for PAYEMS

import matplotlib.pyplot as plt

# Run the Kalman filter/smoother for the last time with optimal parameters
xi, P, predicted_xi, predicted_P = KF_LL(phi_ML, Q_ML, R_ML, y, 0, Q_ML / (1 - phi_ML**2))

T = len(y)

plt.figure(figsize=(10, 6))

# Scatter plot for observations
plt.scatter(range(1, T+1), y, color='k', label='Observations')

# Plot predicted state estimate over time
plt.plot(range(1, T+1), predicted_xi, 'r-', label='Predicted State Estimate', linewidth=2)

# Plot Kalman filter results after update steps
plt.plot(range(1, T+1), xi, 'k-', label='Kalman Filter after Update Step', linewidth=1)

# Labels and title
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('State (xi)', fontsize=12)
plt.title('Kalman Filter Estimates', fontsize=14)

# Add legend and grid
plt.legend(loc='best')
plt.grid(True)

# Show plot
plt.show()

"""# Q2 c"""

#Kalman Filter with MA component

def KF_LL_ARMA(phi, theta, Q, R, y, mu_init, P_init):
    T = len(y)
    predicted_xi = np.zeros(T)  # State predictions
    predicted_P = np.zeros(T)  # State variance predictions
    xi = np.zeros(T)           # State updates
    P = np.zeros(T)            # State variance updates
    eta = np.zeros(T)          # Residuals (store eta_prev)

    # Initialization of the first state prediction
    predicted_xi[0] = mu_init
    predicted_P[0] = P_init

    #(here eta[0] is eta_0 is 0 and eta_1 is the state -phi*previous_state - theta*previous_eta


    # Initial state update
    xi[0] = predicted_xi[0] + (predicted_P[0] / (predicted_P[0] + R)) * (y[0] - predicted_xi[0])
    P[0] = predicted_P[0] - (predicted_P[0] / (predicted_P[0] + R)) * predicted_P[0]
    eta[0] = xi[0] - phi * mu_init

    # Iterative Kalman filter steps
    for t in range(1, T):
        # Prediction step with MA(1) component
        predicted_xi[t] = phi * xi[t - 1] + theta * eta[t - 1]
        predicted_P[t] = phi**2 * P[t - 1] + (1+theta**2) * Q

        # Update step with MA(1) component
        xi[t] = predicted_xi[t] + (predicted_P[t] / (predicted_P[t] + R)) * (y[t] - predicted_xi[t])
        P[t] = predicted_P[t] - (predicted_P[t] / (predicted_P[t] + R)) * predicted_P[t]

        # Update residual
        eta[t] = xi[t] - phi * xi[t - 1] - theta*eta[t-1]

    return eta, xi, P, predicted_xi, predicted_P

def negative_log_likelihood_LL_ARMA(parameter_vector, y):
    phi = parameter_vector[0]
    theta = parameter_vector[1]
    Q = parameter_vector[2] # for the state equation
    R = parameter_vector[3] # for the observation equation
     # for the MA component

    mu_init = 0
    P_init = Q*(theta**2 + 1) / (1 - phi**2)

    #print("phi:", phi)
    _, _, _, predicted_xi, predicted_P = KF_LL_ARMA(phi, theta, Q, R, y, mu_init, P_init)


    #Calculate the Log-likelihood
    LL = 0
    for t in range(len(y)):
        sigma2 = max(predicted_P[t] + R,1e-6)
        mu = predicted_xi[t]
        LL += -0.5*np.log(2*np.pi*sigma2) - 0.5*((y[t] - mu)**2 / sigma2)

    return -LL

#Series INDPRO

y = MainData_demeaned.iloc[:, 0]

initial_guess = [0.5, 0.5, 30, 15]

# Bounds for phi, Q, R, and theta
bounds = [(-1 + 1e-6, 1 - 1e-6), (-1 + 1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None), ]

options = {
#'maxiter': 1000, #MaxIter
#'maxfun': 1000, #MaxFunEvals
'disp': True, #always for Matlab
#'tol': 1e-8, #TolFun
#'gtol': 1e-8, #TolX
#'method': 'BFGS'  #alternatively L-BFGS-B or SLSQP
}

# Perform the optimization
result = minimize(negative_log_likelihood_LL_ARMA, initial_guess, args=(y,), bounds=bounds, #options=options
                  method='SLSQP', tol=1e-8)

# Extract the ML estimates
phi_ML, theta_ML, Q_ML, R_ML  = result.x
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("theta_ML:", theta_ML)
print("ML_LogL:", -ML_LogL)

# Visualization of results for the ARMA(1,1) model for INDPRO


eta_prev, xi, P, predicted_xi, predicted_P = KF_LL_ARMA(phi_ML, theta_ML, Q_ML, R_ML, y, 0, Q_ML*(theta_ML**2 + 1) / (1 - phi_ML**2))

T = len(y)

plt.figure(figsize=(12, 8))

# Observations and Kalman Filter State Estimates
plt.subplot(2, 1, 1)
plt.scatter(range(1, T+1), y, color='k', label='Observations', s=10)
plt.plot(range(1, T+1), predicted_xi, 'r-', label='Predicted State Estimate', linewidth=2)
plt.plot(range(1, T+1), xi, 'b-', label='Kalman Filter (Updated State)', linewidth=1)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.title('ARMA(1,1) Model: Kalman Filter Estimates', fontsize=14)
plt.legend(loc='best')
plt.grid(True)

# Residuals from MA Component
plt.subplot(2, 1, 2)
plt.plot(range(1, T+1), eta_prev, 'g-', label='Residuals (MA Component)', linewidth=2)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Residuals (eta)', fontsize=12)
plt.title('Eta', fontsize=14)
plt.legend(loc='best')
plt.grid(True)


plt.tight_layout()
plt.show()

#Series RPI
y = MainData_demeaned.iloc[:, 1]

initial_guess = [0.5, 0.5, 30, 15]

# Bounds for phi, Q, R, and theta
bounds = [(-1 + 1e-6, 1 - 1e-6), (-1 + 1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None), ]

options = {
#'maxiter': 1000, #MaxIter
#'maxfun': 1000, #MaxFunEvals
'disp': True, #always for Matlab
#'tol': 1e-8, #TolFun
#'gtol': 1e-8, #TolX
#'method': 'BFGS'  #alternatively L-BFGS-B or SLSQP
}

# Perform the optimization
result = minimize(negative_log_likelihood_LL_ARMA, initial_guess, args=(y,), bounds=bounds, #options=options
                  method='SLSQP', tol=1e-8)

# Extract the ML estimates
phi_ML, theta_ML, Q_ML, R_ML  = result.x
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("theta_ML:", theta_ML)
print("ML_LogL:", -ML_LogL)

# Visualization of results for the ARMA(1,1) model for RPI


eta_prev, xi, P, predicted_xi, predicted_P = KF_LL_ARMA(phi_ML, theta_ML, Q_ML, R_ML, y, 0, Q_ML*(theta_ML**2 + 1) / (1 - phi_ML**2))

T = len(y)

plt.figure(figsize=(12, 8))

# Observations and Kalman Filter State Estimates
plt.subplot(2, 1, 1)
plt.scatter(range(1, T+1), y, color='k', label='Observations', s=10)
plt.plot(range(1, T+1), predicted_xi, 'r-', label='Predicted State Estimate', linewidth=2)
plt.plot(range(1, T+1), xi, 'b-', label='Kalman Filter (Updated State)', linewidth=1)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.title('ARMA(1,1) Model: Kalman Filter Estimates', fontsize=14)
plt.legend(loc='best')
plt.grid(True)

# Residuals from MA Component
plt.subplot(2, 1, 2)
plt.plot(range(1, T+1), eta_prev, 'g-', label='Residuals (MA Component)', linewidth=2)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Residuals (eta)', fontsize=12)
plt.title('Eta', fontsize=14)
plt.legend(loc='best')
plt.grid(True)


plt.tight_layout()
plt.show()

#Series PAYEMS
y = MainData_demeaned.iloc[:, 2]

initial_guess = [0.5, 0.5, 30, 15]

# Bounds for phi, Q, R, and theta
bounds = [(-1 + 1e-6, 1 - 1e-6), (-1 + 1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None), ]

options = {

'disp': True, #always for Matlab
#'tol': 1e-8, #TolFun
#'gtol': 1e-8, #TolX
#'method': 'BFGS'  #alternatively L-BFGS-B or SLSQP
}

# Perform the optimization
result = minimize(negative_log_likelihood_LL_ARMA, initial_guess, args=(y,), bounds=bounds, #options=options
                  method='SLSQP', tol=1e-8)

# Extract the ML estimates
phi_ML, theta_ML, Q_ML, R_ML  = result.x
ML_LogL = result.fun

print("phi_ML:", phi_ML)
print("Q_ML:", Q_ML)
print("R_ML:", R_ML)
print("theta_ML:", theta_ML)
print("ML_LogL:", -ML_LogL)

"""# Q2.d"""

#EM Algorithm
def EM_DynamicFactorModel(y, max_iter=1000): #, tol=1e-6):
    T, n = y.shape
    Lambda = np.ones((n, 1))
    Sigma = 0.2 * np.cov(np.diff(y, axis=0), rowvar=False)
    phi = 0.85
    sigma_eta2 = 1.0

    logL_history = []
    f_filt_mean = []
    for iteration in range(max_iter):
        # E-Step: Kalman filter and smoother to estimate the states
        f_smoothed, f_filt ,P_smoothed,  logL, P_cross = KalmanSmoother(y, Lambda, Sigma, phi, sigma_eta2)

        logL_history.append(logL)
        f_filt_mean.append(np.mean(f_filt))


        # M-Step: Update parameters based on smoothed estimates
        # Update Lambda
        Lambda = (np.linalg.inv(f_smoothed.T @ f_smoothed) @ (f_smoothed.T @ y)).T



        y_pred = (Lambda @ f_smoothed.T).T
        residuals = y - y_pred
        Sigma = np.cov(residuals, rowvar=False)


        f_t = f_smoothed[:-1]
        f_t_plus1 = f_smoothed[1:]
        phi = np.sum(f_t * f_t_plus1) / np.sum(f_t**2)
        sigma_eta2 = np.mean((f_t_plus1 - phi * f_t)**2)


    return Lambda, Sigma, phi, sigma_eta2, f_smoothed, f_filt ,logL_history, f_filt_mean

# Kalman Smoother
def KalmanSmoother(y, Lambda, Sigma, phi, sigma_eta2):
    T, n = y.shape
    f_pred = np.zeros((T,1))
    P_pred = np.zeros((T,1,1))
    f_filt = np.zeros((T,1))
    P_filt = np.zeros((T,1,1))


    # Initial values
    f_pred[0] = 0
    P_pred[0] = 10e6

    logL = 0

    # Forward pass: Kalman filter
    for t in range(T):
        # Prediction step
        if t > 0:
            f_pred[t] = phi * f_filt[t - 1]
            P_pred[t,0,0] = phi**2 * P_filt[t - 1,0,0] + sigma_eta2

            P_pred[t] = (P_pred[t] + P_pred[t].T) /2



        v_t = y[t, :].reshape(-1, 1) - (Lambda @ f_pred[t, :]).reshape(1, -1).reshape(-1, 1)

        S_t = (Lambda @ (P_pred[t] @ Lambda.T)) + Sigma

        S_t = (S_t + S_t.T) / 2

        K_t = (P_pred[t] * Lambda.T) @ np.linalg.inv(S_t)



        # Filtered state and covariance updates
        f_filt[t] = f_pred[t] + (K_t @ v_t)
        P_filt[t] = P_pred[t] - (K_t @ Lambda @ P_pred[t])


        # Log-likelihood contribution
        logL += -0.5 * (np.log(np.linalg.det(S_t* 2 * np.pi)) + v_t.T @ np.linalg.inv(S_t) @ v_t )

    # Backward pass: Kalman smoother
    f_smoothed = np.copy(f_filt)
    P_smoothed = np.copy(P_filt)

    #P_pred_T = phi**2 * P_filt[-1, 0, 0] + sigma_eta2

    P_cross = np.zeros((T - 1, 1, 1))

    for t in reversed(range(T-1)):

        J_t = P_filt[t] * phi @ np.linalg.inv(P_pred[t + 1])  # Smoothing gain
        f_smoothed[t] = f_filt[t] + J_t @ (f_smoothed[t + 1] - f_pred[t + 1])
        P_smoothed[t] = P_filt[t] - J_t @ (P_pred[t + 1] - P_smoothed[t + 1]) @ J_t.T
        P_smoothed[t] = (P_smoothed[t] + P_smoothed[t].T) / 2

        if P_pred[t + 1].ndim < 2:
            P_pred[t + 1] = P_pred[t + 1].reshape(1, 1)
        P_cross[t] = P_smoothed[t + 1] @ np.linalg.inv(P_pred[t + 1]) @ (phi * P_filt[t])

    return f_smoothed, f_filt, P_smoothed, logL, P_cross


y1 = MainData_demeaned.iloc[:, 0].to_numpy()
y2 = MainData_demeaned.iloc[:, 1].to_numpy()
y3 = MainData_demeaned.iloc[:, 2].to_numpy()

y = np.column_stack((y1, y2, y3))


print(y.shape)

#Get Estimated Values
Lambda_EM, Sigma_EM, phi_EM, sigma_eta2_EM, f_smoothed_EM, f_filt, logL_history, f_filt_mean = EM_DynamicFactorModel(y)

# Print results
print("Estimated Lambda:", Lambda_EM)
print("Estimated Sigma:", Sigma_EM)
print("Estimated phi:", phi_EM)
print("Estimated sigma_eta2:", sigma_eta2_EM)

#Difference in Log Likelihood Values
print(abs(logL_history[19]- logL_history[999]))

#Mean of f_filt after 1000 iterations
print(f_filt_mean[-1])

#Visualising the Factors
def plot_factors(y, f_filt, f_smoothed):
    T, n = y.shape


    plt.figure(figsize=(12, 8))


    for i in range(n):
        plt.plot(range(T), y[:, i], label=f"Data Series {i+1}", alpha=0.6)


    plt.plot(range(T), f_filt.squeeze(), label="Filtered Factor", color="black", linestyle="--", linewidth=2)


    plt.plot(range(T), f_smoothed_EM.squeeze(), label="Smoothed Factor", color="red", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Comparison of Data Series with Filtered and Smoothed Common Factors")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_factors(y, f_filt, f_smoothed_EM)

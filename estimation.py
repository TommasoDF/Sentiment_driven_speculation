import numpy as np

# Defining the objective function
def objevtive_function(x, g, b, beta):
    return  g*x[0]*(1/(1+np.exp(beta*((b*x[3] - g*x[1])/(b*x[3] + g*x[1]))))) + b*x[2] * (1-(1/(1+np.exp(beta*((b*x[3] - g*x[1])/(b*x[3] + g*x[1]))))))

def two_type_beta_fixed(x, g, b, beta = 0.5):
    return  g*x[0]*(1/(1+np.exp(beta*((b*x[3] - g*x[1])/(b*x[3] + g*x[1]))))) + b*x[2] * (1-(1/(1+np.exp(beta*((b*x[3] - g*x[1])/(b*x[3] + g*x[1]))))))


def two_model_fractions(x, g, b, beta):
    return 1/(1+np.exp(beta*((b*x[3] - g*x[1])/(b*x[3] + g*x[1]))))

def fundamentalist_fractions(x, g, b, beta):
    f_prof1 = (g*x[2] - x[1])/(g*x[2] + b * x[4] - 3*x[1])
    f_prof2 = (b * x[4]- x[1])/(g*x[2] + b * x[4] - 3*x[1])
    f_prof3 = (0- x[1])/(g*x[2] + b * x[4] - 3*x[1])
    prof1 = f_prof1/(f_prof1 + f_prof2 + f_prof3)
    prof2 = f_prof2/(f_prof1 + f_prof2 + f_prof3)
    prof3 = f_prof3/(f_prof1 + f_prof2 + f_prof3)
    frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac3 = np.exp(beta*prof3)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    return frac1, frac2, frac3

def LSTM_fractions(x, g, b, beta):
    #X0 = pt-1, X1 = pt-2, X2 = pt-3, X3 = Bitsi_t-1, X4 = Bitsi_t-2, X5 = lstm_t-1, X6 = lstm_t-3 
    prof1 = (g*x[2] - x[1])/(g*x[2] + b * x[4] + x[6] - 3*x[1])
    prof2 = (b * x[4]- x[1])/(g*x[2] + b * x[4] +  x[6] - 3*x[1])
    prof3 = (x[6]- x[1])/(g*x[2] + b * x[4]  + x[6] + 3*x[1])
    frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac3 = np.exp(beta*prof3)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    return frac1, frac2, frac3

def fundamentalist_full(x, g, b, beta):
    prof1 = (g*x[2] - x[1])/(g*x[2] + b * x[4] + 0 - 3*x[1])
    prof2 = (b * x[4]- x[1])/(g*x[2] + b * x[4] + 0 - 3*x[1])
    prof3 = (0- x[1])/(g*x[2] + b * x[4] + 0 - 3*x[1])
    frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    return g*x[0] * frac1 + b*x[3] * frac2 

def fundamentalist_fitness_moving_avg(x, g, b, beta):
    #term1: x7, term2: x8, term3: x9, terms4: x10, term5: x11, term6: x12
    prof1 = g*x[7] + x[11] - g*x[8] - x[12]
    prof2 = b*x[9] + x[11] - b*x[10] - x[12]
    prof3 = x[11] - x[12]
    frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    return g*x[0] * frac1 + b*x[3] * frac2 

def fundamentalist_beta_fixed(x, g, b, beta = 0.1):
    prof1 = (g*x[2] - x[1])/(g*x[2] + b * x[4] - 3*x[1])
    prof2 = (b * x[4]- x[1])/(g*x[2] + b * x[4] - 3*x[1])
    prof3 = (0- x[1])/(g*x[2] + b * x[4] - 3*x[1])
    #frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac1 = 1/(1 + np.exp(beta*(prof2-prof1)) + np.exp(beta*(prof3-prof1)))
    #frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = 1/(np.exp(beta*(prof1-prof2)) + 1 + np.exp(beta*(prof3-prof2)))
    return g*x[0] * frac1 + b*x[3] * frac2

def LSTM_full(x, g, b, beta):
    #X0 = pt-1, X1 = pt-2, X2 = pt-3, X3 = Bitsi_t-1, X4 = Bitsi_t-2, X5 = lstm_t-1, X6 = lstm_t-3 
    prof1 = (g*x[2] - x[1])/(g*x[2] + b * x[4] + x[6] - 3*x[1])
    prof2 = (b * x[4]- x[1])/(g*x[2] + b * x[4] +  x[6] - 3*x[1])
    prof3 = (x[6]- x[1])/(g*x[2] + b * x[4]  + x[6] + 3*x[1])
    frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac3 = np.exp(beta*prof3)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    return g*x[0] * frac1 + b*x[3] * frac2 + x[5] * frac3

def LSTM_beta_fixed(x, g, b, beta = 0.1):
    #X0 = pt-1, X1 = pt-2, X2 = pt-3, X3 = Bitsi_t-1, X4 = Bitsi_t-2, X5 = lstm_t-1, X6 = lstm_t-3 
    prof1 = (g*x[2] - x[1])/(g*x[2] + b * x[4] + x[6] - 3*x[1])
    prof2 = (b * x[4]- x[1])/(g*x[2] + b * x[4] +  x[6] - 3*x[1])
    prof3 = (x[6]- x[1])/(g*x[2] + b * x[4]  + x[6] + 3*x[1])
    frac1 = np.exp(beta*prof1)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac2 = np.exp(beta*prof2)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    frac3 = np.exp(beta*prof3)/(np.exp(beta*prof1) + np.exp(beta*prof2) + np.exp(beta*prof3))
    return g*x[0] * frac1 + b*x[3] * frac2 + x[5] * frac3

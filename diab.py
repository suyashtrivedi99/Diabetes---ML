"""
Objective: -    Train a logistic regressional model on a diabetes dataset,
                and predict whether a patient is diabetic or not        
"""

import numpy as np 
import scipy.optimize as op
import sklearn.model_selection as ms
import matplotlib.pyplot as plt

def sigmoid(X):                         
    return 1 / (1 + np.exp(-X))

def cost_func(theta, X, y, lamb):   #Cost function
    
    m = X.shape[0]
    z = np.dot(X ,theta)
    h = sigmoid(z)
    
    term1 = (-y) * np.log(h) 
    term2 = (1 - y) * np.log(1 - h)
    
    J = (1/m) * np.sum(term1 - term2)
    J = J + (lamb/(2 * m)) * (np.sum(theta * theta)   #Regularization term        
        - (theta[0] * theta[0]))                                    
    
    return J


def grad(theta, X, y, lamb):    #Gradient function
    
    m = X.shape[0]
    z = np.dot(X, theta)
    h = sigmoid(z)
    
    diff = h - y
    mat = np.dot(X.T ,diff)
    
    G = (1/m) * mat

    temp = G[0]                     #Not regularizing the constant term
    G = G + ((lamb/m) * theta)      #Regularization term
    G[0] = temp
        
    return G


fname = "diabetes.csv"                      
data = np.loadtxt(fname, delimiter=",")  #Read dataset into an nparray  

m, n = data.shape

X = data[:, 0:n-1]  #Feature matrix
y = data[:, n-1]    #Target variable

# Plotting a bar-graph for number of patients vs outcome
 
label = ['Not Diabetic', 'Diabetic']
no_of_patients = [m - np.sum(y), np.sum(y)]
index = [0, 1]
plt.bar(index, no_of_patients)
plt.xlabel('Outcome')
plt.ylabel('Number of patients')
plt.xticks(index, label, rotation=30)
plt.title('Number of patients having/not having Diabetes')
plt.show()


meanX = np.mean(X, axis = 0)    
stdX = np.std(X, axis = 0)

for i in range(n-1):
    X[:, i] = (X[:, i] - meanX[i])/(stdX[i])    #Feature scaling

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.33, random_state = 200) #Dividing data into training and testing part

lamb = 100      #Regularization constant
m ,n = X.shape
initial_theta = np.zeros(n)

Result = op.minimize(fun = cost_func, x0 = initial_theta,
                     args = (X_train, y_train, lamb), method = 'TNC', jac = grad) #Minimizing Cost Function 

opt_theta = Result.x
print("Optimal Parameters obtained: \n", opt_theta)


y_pred = np.dot(X_test, opt_theta)

err = 0
nerr = 0

m1 = y_pred.shape[0]

for i in range(m1):
    diff = y_pred[i]
    
    if (diff >= 0.5):
        y_pred[i] = 1
        
    else:
        y_pred[i] = 0
    
    if (y_pred[i] != y_test[i]):
        err = err + 1
        
    else:
        nerr = nerr + 1
        
#Plotting contribution of each feature in detemining outcome

label = ['Pregnancies', 'Glucose', 'BP', 'SkinThickness',
         'Insulin', 'BMI', 'DPF', 'Age']
index = np.arange(len(label))
plt.bar(index, opt_theta)
plt.xlabel('Feature')
plt.ylabel('Weight of Feature')
plt.xticks(index, label, rotation=30)
plt.title('Relative contribution of each feature in determining outcome')
plt.show()

print("\n Conclusions from the graph: \n")
print("- Glucose, Pregnancies and BMI have significant contribution in determining diabetes\n")
print("- DiabetesPedigreeFunction and age have moderate contribution\n")
print("- Skin Thickness and Insulin have negligible contribution\n")
print("- BP has negative contribution i.e lower values of BP favour diabetes more than higher values\n")

print("\n Accuracy Obtained (%): \n", (nerr/(nerr+err))*100)    

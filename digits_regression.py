


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Sigmoid function
def logistic(x):
  return 1.0 / (1.0 + np.exp(-1.0*x))


def hypothesisLogistic(X, coefficients, bias):
    
    # array of zeros. Length is same as number of training rows.  
    predictedValues = np.zeros(X.shape[0])
    
    # for each feature multiple the X training data by the appropriate 
    # coefficient and add to to predictedvalues
    for num in range(len(coefficients)):
        predictedValues += coefficients[num] * X[:, num]
    
    # finally add the current bias to each of the predicted values
    predictedValues += bias
    
    logisticPredicitons = logistic(predictedValues)
    
    return logisticPredicitons


def gradient_descent_log(bias, coefficients, alpha, X, Y, max_iter):

    length = len(Y)
    
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    for num in range(0, max_iter):
        
        # Calculate predicted y values for current coefficient and bias values 
        predictedY = hypothesisLogistic(X, coefficients, bias)
        

        # calculate gradient for bias
        biasGrad =    (1.0/length) *  (np.sum( predictedY - Y))
        
        #update bias using GD update rule
        bias = bias - (alpha*biasGrad)
        
        # for loop to update each coefficient value in turn
        for coefNum in range(len(coefficients)):
            
            # calculate the gradient of the coefficient
            gradCoef = (1.0/length)* (np.sum( (predictedY - Y)*X[:, coefNum]))
            
            # update coefficient using GD update rule
            coefficients[coefNum] = coefficients[coefNum] - (alpha*gradCoef)
        
        # Cross Entropy Error 
        cost = np.average(((-1*Y)*(np.log(predictedY)))- ((1-Y)*(np.log(1-predictedY))))

        errorValues.append(cost)
    
    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients


def calculateAccuracy(bias, coefficients, X_test, y_test):
    
    # Get all predicted values for the text set
    predictedYValues = hypothesisLogistic(X_test, coefficients, bias)
   
    # Logistic regression is a probabilistic classifier.
    # If the probability is less than 0.5 set class to 0
    # If probability is greater than 0.5 set class to 1 
    predictedYValues[predictedYValues <= 0.5] = 0
    predictedYValues[predictedYValues > 0.5] = 1

    print ("Final Accuracy: ", sum(predictedYValues==y_test)/len(y_test))
    

def logisticRegression(X_train, y_train, X_test, y_test):

    # set the number of coefficients equal to the number of features
    coefficients = np.zeros(X_train.shape[1])
    bias = 0.0
   
    alpha = 0.005 # learning rate
    
    max_iter=1000

    # call gredient decent, and get intercept(bias) and coefficents
    bias, coefficients = gradient_descent_log(bias, coefficients, alpha, X_train, y_train, max_iter)
    
    calculateAccuracy(bias, coefficients, X_test, y_test)
    


def main():
    
    digits = datasets.load_digits()
    
    # Display one of the images to the screen
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    
    # Load the feature data and the class labels
    X_digits = digits.data
    y_digits = digits.target
    
    # The logistic regression model will differentiate between two digits
    # Code allows you specify the two digits and extract the images 
    # related to these digits from the dataset
    indexD1 = y_digits==0 
    indexD2 = y_digits==1
    allindices = indexD1 | indexD2
    X_digits = X_digits[allindices]
    y_digits = y_digits[allindices]
    
    # Standarize the data
    scaler = preprocessing.StandardScaler()
    X_digits = scaler.fit_transform(X_digits)

 
    n_samples = len(X_digits)
    
    # Training data 
    X_train = X_digits[:int(.9 * n_samples)]
    y_train = y_digits[:int(.9 * n_samples)]
    
    # Test data
    X_test = X_digits[int(.9 * n_samples):]
    y_test = y_digits[int(.9 * n_samples):]

   
    logisticRegression(X_train, y_train, X_test, y_test)
    
    
  

main()

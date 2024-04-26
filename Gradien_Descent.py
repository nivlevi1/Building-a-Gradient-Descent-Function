#!pip install pandas
#!pip install numpy
#!pip install seaborn
#!pip install matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/nivlevi1/Building-a-Gradient-Descent-Function/main/sample.csv'
df = pd.read_csv(url)

def gradient_descent(df, intercept=0, slope=0, min_step_size=0.0001, steps=100000):
    #Save our Parameters :
    inter_values = []
    slope_values = []
    loss_func_values = []
    
    
    # Set the learning rate
    learning_rate = 0.0001
    steps_count = 0 # Initialize the step count
    
    # Get the x and y values from the input dataframe
    x = df['x'].values
    y = df['y'].values
    
    # Loop through the specified number of steps
    for i in range(steps):
        # Calculate the predicted y values based on the current intercept and slope
        Y_pred = intercept + slope * x
        
        # Calculate the partial derivative of the SSE with respect to the intercept and slope
        loss_func = np.sum((y - Y_pred)**2)/len(x)
        d_SSE_intercept = -2 * np.sum(y - Y_pred)/len(x)
        d_SSE_slope = -2 * np.sum(x * (y - Y_pred))/len(x)
        
        # Calculate the step size for the intercept and slope
        step_size_intercept = d_SSE_intercept * learning_rate
        step_size_slope = d_SSE_slope * learning_rate
        
        # Update the intercept and slope based on the step sizes
        intercept -= step_size_intercept
        slope -= step_size_slope
        
        steps_count += 1
        
        #Save the parameters in order to generate the plots
        inter_values.append(intercept)
        slope_values.append(slope)
        loss_func_values.append(loss_func)
        
        
        # Check if the step sizes are smaller than the specified minimum step size
        # If the step sizes are small enough, return the current intercept, slope, and step count
        # If the maximum number of steps is reached, return the current intercept, slope, and step count
        
        if abs(step_size_intercept) < min_step_size and abs(step_size_slope) < min_step_size:
            return intercept, slope, steps_count, inter_values, slope_values, loss_func_values
        
    return intercept, slope, steps_count, inter_values, slope_values, loss_func_values

gradients_values = gradient_descent(df)
print(f'intercept: {gradients_values[0]}, Slope: {gradients_values[1]}, Steps : {gradients_values[2]}')


def main() :
    intercepts = gradients_values[3]
    slopes = gradients_values[4]
    loss_values = gradients_values[5]
    
    
    # create figure with three subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
    
    
    ax=axs[0,0].plot(intercepts, 'tab:purple')
    axs[0,0].set_xlabel("Step")
    axs[0,0].set_ylabel("Intercept")
    axs[0,0].set_title('Intercept VS step')
    
    
    ax=axs[0,1].plot(slopes, 'tab:green')
    axs[0,1].set_xlabel("Steps")
    axs[0,1].set_ylabel("Slope")
    axs[0,1].set_title('Slope VS step')
    
    
    ax=axs[1,0].plot(loss_values, 'tab:red')
    axs[1,0].set_xlabel("Steps")
    axs[1,0].set_ylabel("Loss_func_value")
    axs[1,0].set_title('loss Function VS step')
    
    
    ax=axs[1,1].plot(loss_values[2:], 'tab:red')
    axs[1,1].set_xlabel("Steps")
    axs[1,1].set_ylabel("Loss_func_value")
    axs[1,1].set_title('loss Function VS step - Zoom in ')
    
    
    # show plot
    fig.tight_layout() 
    plt.show()


#### main()

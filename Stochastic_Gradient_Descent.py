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

def stoch_gradient_descent(df, intercept=0, slope=0, min_step_size=0.0001, steps=1000):
    #Save our Parameters :
    inter_values = []
    slope_values = []
    loss_func_values = []
    
    learning_rate = 0.0001
    steps_count = 0
    for i in range(steps):
        sample = df.sample(1) #We should take different sample point for each step
        x = sample['x'].values
        y = sample['y'].values
        Y_pred = intercept + slope * x
        
        loss_func = np.sum((y - Y_pred)**2)/len(x)
        d_SSE_intercept = -2 * (y - Y_pred)
        d_SSE_slope = -2 * (x * (y - Y_pred))
        
        step_size_intercept = d_SSE_intercept * learning_rate
        step_size_slope = d_SSE_slope * learning_rate
        
        intercept -= step_size_intercept
        slope -= step_size_slope
        
        steps_count += 1
        
        inter_values.append(float(intercept))
        slope_values.append(float(slope))
        loss_func_values.append(loss_func)
        
        if abs(step_size_intercept) < min_step_size and abs(step_size_slope) < min_step_size:
            return float(intercept), float(slope), steps_count, inter_values, slope_values, loss_func_values
        
                     
    return float(intercept), float(slope), steps_count, inter_values, slope_values, loss_func_values

stoch_gradients_values = stoch_gradient_descent(df)
print(f'intercept: {stoch_gradients_values[0]}, Slope: {stoch_gradients_values[1]}, Steps : {stoch_gradients_values[2]}')




def main():
    intercepts = stoch_gradients_values[3]
    slopes = stoch_gradients_values[4]
    loss_values = stoch_gradients_values[5]
    
    
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
    
    
    ax=axs[1,1].plot(loss_values[7:], 'tab:red') 
    axs[1,1].set_xlabel("Steps")
    axs[1,1].set_ylabel("Loss_func_value")
    axs[1,1].set_title('loss Function VS step - Zoom in ')
    
    
    # show plot
    fig.tight_layout() 
    plt.show()

#main()
        

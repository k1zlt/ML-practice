import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/Linear Regression - Sheet1.csv')

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i]['X']
        y = points.iloc[i]['Y']
        
        m_gradient += - (2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += - (2 / n) * (y - (m_now * x + b_now))
    
    return m_now - m_gradient * L, b_now - b_gradient * L

m = 0
b = 0
L = 0.00001
epochs = 100

for i in range(epochs):
    print("Epochs:", i, "m:", m, "b:", b)
    m, b = gradient_descent(m, b, df, L)
    
plt.scatter(df['X'], df['Y'], label='Data Points')
x_range = range(int(df['X'].min()), int(df['X'].max()) + 1)
plt.plot(x_range, [m * x + b for x in x_range], color="black", label='Best Fit Line')
plt.legend()
plt.show()
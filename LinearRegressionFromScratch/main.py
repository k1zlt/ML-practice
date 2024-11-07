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
        
        m_gradient += - 2 / n * x * (y - (m_now * x + b_now))
        b_gradient += - 2 / n * (y - (m_now * x + b_now))
    
    return m - m_gradient * L, b - b_gradient * L

m = 0
b = 0
L = 0.0001
epochs = 100

for i in range(epochs):
    print("Epochs:", i, "m:", m, "b:", b)
    m, b = gradient_descent(m, b, df, L)
    
print(m, b)
plt.scatter(df['X'], df['Y'])
plt.plot(list(range(300)), [m * x + b for x in range(300)], color="black")
plt.show()
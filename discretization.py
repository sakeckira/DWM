import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample continuous data
data = {
    'Age': [15, 22, 25, 36, 42, 52, 67, 72, 81, 90]
}
df = pd.DataFrame(data)

# 1. Discretization using bins
bins = [0, 20, 40, 60, 80, 100]
labels = ['Teen', 'Young Adult', 'Adult', 'Senior', 'Elder']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# 2. Display discretized data
print(df)

# 3. Visualization using bar chart
df['Age_Group'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

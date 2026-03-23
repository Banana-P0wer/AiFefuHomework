import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
values_df = pd.read_csv("/Users/vlad/Developer/Oskin's labs/First/Pokemon.csv", delimiter=',')

# Clean data and convert to numeric format
values_df['Height'] = values_df['Height'].str.extract(r'(\d+\.\d+)').astype(float)
values_df['Weight'] = values_df['Weight'].str.extract(r'(\d+\.\d+)').astype(float)
values_df['HP Base'] = values_df['HP Base'].astype(float)

# Group data by height and calculate the mean value for each group
average_values_by_height = values_df.groupby('Height').agg({'Weight': 'mean', 'HP Base': 'mean'}).reset_index()

# Visualization
plt.figure()

# Plot sorted height values in green
sorted_height = values_df['Height'].sort_values().unique()
plt.plot(sorted_height, 'g', label='Sorted Height')

# Plot average weight and base HP values
plt.plot(np.arange(len(average_values_by_height['Weight'])), average_values_by_height['Weight'], 'b', label='Average Weight')
plt.plot(np.arange(len(average_values_by_height['HP Base'])), average_values_by_height['HP Base'], 'r', label='Average HP Base')

plt.legend()
plt.xlabel('Unique Heights')
plt.ylabel('Values')
plt.title('Height and Average Values Comparison')

plt.show()

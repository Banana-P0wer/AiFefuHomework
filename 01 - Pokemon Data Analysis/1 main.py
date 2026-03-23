import pandas as pd

values_df = pd.read_csv("/Users/vlad/Developer/Oskin's labs/First/Pokemon.csv", delimiter=',')

print('\nGeneral information about the table data')
print(values_df.info())

print('\nTable data output')
print(values_df)

print('\nColumn names output')
print(values_df.columns)

print('\nNumber of missing values')
print(values_df.isnull().sum())

print(f'\nNumber of duplicate rows: {values_df.duplicated().sum()}')


# Clean data and convert to numeric format
#values_df['Height'] = values_df['Height'].str.extract(r'(\d+\.\d+)').astype(float)
#values_df['Weight'] = values_df['Weight'].str.extract(r'(\d+\.\d+)').astype(float)
#values_df['HP Base'] = values_df['HP Base'].astype(float)


# Sort by the 'Height' column
values_df.sort_values(by=['Height', 'Weight', 'HP Base'], inplace=True)

# Show results
print(values_df.head())


print('Feature statistics')
print(values_df.describe())

print('\nFirst 10 table values')
print(values_df.head(10))


import numpy as np

values_arr_Height = np.array(values_df)[:, 3]
values_arr_Weight = np.array(values_df)[:, 4]
values_arr_HP_Base = np.array(values_df)[:, 14]

print('Array data output')
print(values_arr_Height)

import matplotlib
import matplotlib.pyplot as plt
from pylab import plot

matplotlib.style.use('ggplot')


plt.figure()

x = np.linspace(values_arr_Height[0], values_arr_Height[values_arr_Height.size - 1], values_arr_Height.size)

plt.plot(x, values_arr_Height,  'g', label='Height')
#plt.plot(x, values_arr_Weight,  'b', label='Weight')
#plt.plot(x, values_arr_HP_Base, 'r', label='HP Base')

plt.legend()
plt.xlabel('Height')
plt.ylabel('Values')
plt.title('Height and Average HP Base Comparison')

plt.show()

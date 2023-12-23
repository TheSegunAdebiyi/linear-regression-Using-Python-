# linear-regression-Using-Python
Analyzing and Predicting Adjusted Net National Income per Capita in Nigeria: A Python Data Science Journey
### Project Overview 

In the realm of data science, understanding and predicting economic indicators play a pivotal role in decision-making processes. In this project, we delved into the intricacies of Nigeria's economic landscape by exploring the Adjusted Net National Income (ANNI) per capita. Employing Python, along with libraries such as pandas, seaborn, matplotlib, and sklearn, we embarked on a comprehensive analysis and prediction journey.
### Data Source 
World Bank staff estimates based on sources and methods in World Bank's "The Changing Wealth of Nations: Measuring Sustainable Development in the New Millennium" ( 2011 ).
[Download here](https://api.worldbank.org/v2/en/indicator/NY.ADJ.NNTY.PC.CD?downloadformat=csv)
### Data Collection and Cleaning
- We kickstarted the project by sourcing relevant data on Nigeria's Adjusted Net National Income per Capita. This involved gathering historical data to establish a robust foundation for our analysis.
Leveraging pandas. 
- we meticulously cleaned the data by handling missing values, outliers, and ensuring data integrity. This step is crucial for accurate modeling and forecasting.

```python
import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'Adjusted net national income per capita (current US$) - Nigeria - Sheet1.csv'
df = pd.read_csv(file_path)

# Remove commas from the "Value" column and convert it to float
df['Value'] = df['Value'].str.replace(',', '').astype(float)

# Display the updated DataFrame
df
# Save the DataFrame to a CSV file
df.to_csv('output_file.csv', index=False)

```
### Exploratory Data Analysis (EDA)
-Seaborn and matplotlib were employed to visualize the distribution of ANNI per capita over time. We created insightful visualizations, such as line plots and histograms, to uncover trends, patterns, and potential anomalies within the data.
-Correlation matrices and heatmaps aided in identifying relationships between ANNI per capita and other relevant variables, providing valuable insights for the regression model.

```python
pip install scikit-learn
pip install pandas matplotlib seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Linear Regression Modeling
-With a clean dataset in hand, we utilized the scikit-learn library to implement a linear regression model. This involved splitting the data into training and testing sets to assess the model's performance accurately.
-We explored the coefficients and intercept to understand the impact of independent variables on the Adjusted Net National Income per Capita.
### Model Evaluation
Employing metrics like Mean Squared Error (MSE) and R-squared, we evaluated the performance of our linear regression model. This step ensured that our model accurately captured the underlying patterns in the data and could be relied upon for predictions.
### Prediction for 2021
Utilizing the trained model, we predicted the Adjusted Net National Income per Capita for the year 2021. This predictive analysis is crucial for anticipating economic trends and making informed decisions.
### Visualization of Predictions
Seaborn and matplotlib were once again instrumental in visually representing our model's predictions. We created visually appealing plots, overlaying predicted values on the actual data, to provide a clear picture of the model's efficacy.


```python
from sklearn.linear_model import LinearRegression
data = pd.read_csv('output_file.csv')
data.info()
sns.regplot(x='Year', y='Value', data=data)
# regression
# creating instance
lri = LinearRegression()
# trainning model
lri.fit(data[['Year']].values, data.Value)
lri.predict([[2021]])
```
![Screenshot 2023-12-19 050849](https://github.com/TheSegunAdebiyi/linear-regression-Using-Python-/assets/107259515/626f1195-f83a-46e3-a9e9-5a9145b015ed)

### Conclusion 

In conclusion, our data science journey through Nigeria's Adjusted Net National Income per Capita showcased the power of Python libraries in extracting meaningful insights. The linear regression model not only provided accurate predictions but also highlighted the key factors influencing the economic indicator.

This project underscores the importance of data-driven decision-making in economic analysis and showcases the versatility of Python for such endeavors.


Moving forward, the model can be enhanced by incorporating additional variables and exploring advanced regression techniques. Moreover, ongoing data collection and analysis will contribute to refining the model's accuracy and reliability.

In summary, our exploration of Adjusted Net National Income per Capita in Nigeria serves as a testament to the capabilities of Python in the field of data science. This project not only provides valuable insights into Nigeria's economic landscape but also lays the groundwork for future endeavors in predictive analytics.



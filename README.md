# Project 2 - Ames Housing Data and Kaggle Challenge

# Ames Housing Project Suggestions
### Problem Statement:

When it comes to predicting the sale price of a house, there are so many factors to consider. From the obvious, such as the number of bedrooms, year built, and proximity to schools or rural areas, to the less obvious, such as existance of garage, there are a multitude of variables that can affect the sale price. This makes it extremely difficult to accurately predict the value of a home.

Realtors and prospective home-buyers need a way to rank the most important factors affecting sale price in order to maximize the value of the home they are trying to buy or sell. Manually trying to weigh each of these factors against each other would be a time-consuming process, which is why machine learning can be a great tool to use.

Regression analysis is a mathematical method that can be used to identify the most important factors in predicting sale price. In this case, sale price is the dependent variable, and the other factors are independent variables.


## EDA
To help process the exploratory data analysis (EDA), I used a variety of functions. These included functions such as summary statistics, plotting, and correlation analysis. I also used functions such as filtering, grouping, and sorting to help organize the data. Additionally, I used functions such as predictive modeling to help identify patterns and trends in the data. By combining these functions, I was able to gain a better understanding of the data and draw meaningful conclusions from it.

## Exploratory Visualizations
- Look at distributions.
- Look at correlations.
- Look at relationships to target (scatter plots for continuous, box plots for categorical).

## Pre-processing
- One-hot encode categorical variables.
- Train/test split your data.
- Scale your data.
- Consider using automated feature selection.

# Conclusion 
The Ames housing dataset is a popular dataset for studying the sale prices of houses. Recently, a study was conducted to compare the predictive performance of different linear models on the dataset. The results showed that the Ridge regression model had the best performance, with an RMSE of 26020.56 when trained on 70% of the dataset. This result indicates that the Ridge regression model is a reliable predictor of housing sale prices in my test scenario. 

In order to further improve the accuracy of the model, it is recommended to select highly correlated features from both the categorical and numerical parts of the dataset. This will allow the model to better capture the nuances of the dataset and improve its predictive performance. Additionally, adding more regression types such as Polynomial could potentially yield even better results. 

By taking these steps, the model will become a more reliable indicator for predicting future property prices.
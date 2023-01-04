# Project 2 - Ames Housing Data and Kaggle Challenge

# Ames Housing Project Suggestions
### Problem Statement:

When it comes to predicting the sale price of a house, there are so many factors to consider. From the obvious, such as the number of bedrooms, year built, and proximity to schools or rural areas, to the less obvious, such as existance of garage, there are a multitude of variables that can affect the sale price. This makes it extremely difficult to accurately predict the value of a home.

Realtors and prospective home-buyers need a way to rank the most important factors affecting sale price in order to maximize the value of the home they are trying to buy or sell. Manually trying to weigh each of these factors against each other would be a time-consuming process, which is why machine learning can be a great tool to use.

Regression analysis is a mathematical method that can be used to identify the most important factors in predicting sale price. In this case, sale price is the dependent variable, and the other factors are independent variables.


## EDA
To help process the exploratory data analysis (EDA), I used a variety of functions. These included functions such as summary statistics, plotting, and correlation analysis. I also used functions such as filtering, grouping, and sorting to help organize the data. Additionally, I used functions such as predictive modeling to help identify patterns and trends in the data. By combining these functions, I was able to gain a better understanding of the data and draw meaningful conclusions from it.

## Data Cleaning
Detail about each columns that shows the number of missing values and unique values.
| Columns |dtype |null | unique |
|id	int64 |	0 | 878 |
|pid | int64 | 0 | 878 | 
|ms_subclass | int64 | 0 | 15 |
|lot_frontage | float64 | 160 | 104 |
|lot_area | int64 | 0 | 89 |
|street | object |0 | 2 |
|lot_shape | object | 0 | 4 |
|land_contour | object | 0 | 4 |
|lot_config | object | 0 | 5 |
|land_slope | object | 0 | 3 |
|condition_1 | object | 0 | 9 |
|bldg_type | object | 0 | 5 |
|house_style | object | 0 | 8 |
|overall_qual | int64 | 0 | 9 |
|overall_cond | int64 | 0 | 9 |
|year_built | int64 | 0 | 106 |
|year_remodadd | int64 | 0 | 61 |
|mas_vnr_area | float64 | 1 | 232 |
|exter_qual| object | | 0 | 4 |
|exter_cond | object | | 0 | 5 |
|foundation | object | 0 | 6 |
|bsmt_qual | object | 25 | 5 |
|bsmt_exposure | object | 25 | 4 |
|bsmtfin_type_1 | object | 25 |6 |
|bsmtfin_sf_1 | int64 | 0 | 462 |
|bsmtfin_type_2 | object | 25 | 6 |
|bsmtfin_sf_2 | int64 | 0 | 96 |
|bsmt_unf_sf | int64 | 0 | 562 |
|total_bsmt_sf | int64 | 0 | 526 |
|central_air | object | 0 | 2 |
|electrical | object | 1 | 4 |
|st_flr_sf | int64 | 0 | 560 |
|2nd_flr_sf | int6| 4 | 0 | 285 |
|low_qual_fin_sf | int64 | 0 | 8 |
|gr_liv_area | int64 | 0 | 621 |
|bsmt_full_bath | int64 | 0 | 3 |
|bsmt_half_bath | int64 | 0 | 2 |
|full_bath | int64 | 0 | 5 |
|half_bath | int64 | 0 | 3 |
|bedroom_abvgr | int64 | 0 | 7 |
|kitchen_abvgr | int64 | 0 | 4 |
|totrms_abvgrd | int64 | 0 | 10 |
|fireplaces | int64 | 0 | 4 |
|garage_type | object | 44 | 6 |
|garage_finish | object | 45 |3 |
|garage_cars | int64 |0 | 5 |
|garage_area | int64 | 0 | 357 |
|garage_cond | object | 45 | 5 |
|paved_drive | object | 0 | 3 |
|wood_deck_sf | int64 | 0 | 210 |
|open_porch_sf | int64 | 0 | 162 |
|enclosed_porch | int64 | 0 | 83|
|3ssn_porch	int64 | 0 | 12 |
|screen_porch | int64 | 0 | 54 |
|pool_area | int64 | 0 | 5 |
|misc_val | int64 | 0 | 21 |
|mo_sold | int64 | 0 | 12 |
|yr_sold |int64 | 0 | 5 |

Number of categorical column :21 | Columns with Null: 7 | Total number of categorical Null values: 565
========
Column: street | Null: False| Null Count: 0 | Unique : 2
Column: lot_shape | Null: False| Null Count: 0 | Unique : 4
Column: land_contour | Null: False| Null Count: 0 | Unique : 4
Column: lot_config | Null: False| Null Count: 0 | Unique : 5
Column: land_slope | Null: False| Null Count: 0 | Unique : 3
Column: condition_1 | Null: False| Null Count: 0 | Unique : 9
Column: bldg_type | Null: False| Null Count: 0 | Unique : 5
Column: house_style | Null: False| Null Count: 0 | Unique : 8
Column: exter_qual | Null: False| Null Count: 0 | Unique : 4
Column: exter_cond | Null: False| Null Count: 0 | Unique : 5
Column: foundation | Null: False| Null Count: 0 | Unique : 6
Column: bsmt_qual | Null: True| Null Count: 55 | Unique : 6
Column: bsmt_exposure | Null: True| Null Count: 58 | Unique : 5
Column: bsmtfin_type_1 | Null: True| Null Count: 55 | Unique : 7
Column: bsmtfin_type_2 | Null: True| Null Count: 56 | Unique : 7
Column: central_air | Null: False| Null Count: 0 | Unique : 2
Column: electrical | Null: False| Null Count: 0 | Unique : 5
Column: garage_type | Null: True| Null Count: 113 | Unique : 7
Column: garage_finish | Null: True| Null Count: 114 | Unique : 4
Column: garage_cond | Null: True| Null Count: 114 | Unique : 6
Column: paved_drive | Null: False| Null Count: 0 | Unique : 3

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
we started by removing null values form our targets that is RainToday and RainTomorrow
we import sklearn.model_selection 
of train_test_split to split data for the work
we made some portion of train_val_data and created a test dataframe 80 and 20 for test
then we split the train_val_data to val_df,train_df 75 for the train for the 75 and 25 for the val_df datafram which means you will train the 75 percent of the 
data and you check whether is working with the val_datafram of 
we will create it will for the data frame
now we make a categorical columns for the data and the numerical columns for the data and we make it to be in our training dataset,val and test.
After that we remove all the nan values by using a machine learning 
module remover SimpleImputer from sklearn
we fit the data into the Imputer using a strategy of either mean or median.
After fitting then we transfor the data.
now we have to scale the values into minmax meaning each values in the data will be range from zero to one using th MinMaxScaler function form sklearn to make all the values 0 and 1

now we use the OneHotEncoding from preprocessing to split the data in respective of the of the categorical columns. we also use the get_features_name_out to make each unique for the row data 
## After that you save using parquet format 
**We therefore use the saved files to make the Logical regression**
>> using linear_model for the LogicalRegressions
## we will then check the accuracy 85%

## [confusion_matrix] was used to make us verify
True positive or True negatibe
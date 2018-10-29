# ML_project1
The first machine learning project
## Files
1. run.py
It is the main file of our project. By running this file, you can choose different models and different processes by the answer 0/1 from the keyboard to the demand asked in the console. As discussed in the report, we have 2 models: One is the light model that we apply the feature selection to the raw data, while the other one is the full model that contains all the features from the raw data. 
For each model, we can choose if or not to generate the best hyperparameters by cross validtion. If we skip the cross validation, then the best hyper parameters will be generated by the scores. We can also choose if use this parameters to train the model and generate the predictions. If yes, then one submission file for each model will be generated, which is the same as we submit on the kaggle. And the best score on the kaggle comes from the full model.

2. preprocessing.py
This contains the preprocessing functions : standardization, replace_invalid_value

3. feature_engineering.py
It contains the operations we do to select or extend the features, and a function with the same name that combines all the operations together, by taking an class "ops" as an input, to tell this function to execute which operations in which order.

4. cross_validation.py
It contains the function to split the data and the cross validation functions for both ridge regression and logistic regression, returning the mean value of the scores.

5. Other basic files: implementations.py, cost_functions, gradients.py, project_func.py, proj1_helpers.py
## contributers

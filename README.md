
# [Higgs Boson Challenge][kaggle]

[Kaggle competition][kaggle] between EPFL Machine Learning course (CS433) students

Goals: (see description pdf for full details)
Results: [Ranked 82th, with accuracy = 0.82512][leaderboard] (teamname = 0 error(s), 0 warning(s)) out of ~150 teams (400+ EPFL students)
## Folders
0. README
1. data: It contains the raw training and testing data that we download from the [Kaggle Higgs Boson Challenge][kaggle].
2. scripts: It contains the python code files
3. results: We store the final submission csv file here.

## To generate the submission
1. Download the train and test data set, and put them in to a folder named data, which is at the same level as the folder scripts.
2. Run the file run.py
3. Attention: after click the run file botton, please do not just left the program running aside, since their will be the messges displaying on the console, and require the input from the keyboard. Pay attention to what will be shown on the console and reply to the questions according to your needs. If you don't have the scores.npy and scores_reduced.npy in the scripts folder, please do not choose to skip the cross validation steps.
4. The first question occurs to you will be choosing the light data model or the full data model, which correspond respectively to the data with the feature selection and without the feature selection.
5. The final submission csv files will be generated in the case that you choose to run the training and prediction steps. The submission csv files will be stored in the folder named results.

## Scripts
1. run.py </li>It is the main file of our project. By running this file, you can choose different models and different processes by the answer 0/1 from the keyboard to the demand asked in the console. As discussed in the report, we have 2 models: One is the light model that we apply the feature selection to the raw data, while the other one is the full model that contains all the features from the raw data. </li>For each model, we can choose if or not to generate the best hyperparameters by cross validtion. If we skip the cross validation, then the best hyper parameters will be generated by the scores. We can also choose if use this parameters to train the model and generate the predictions. If yes, then one submission file for each model will be generated, which is the same as we submit on the kaggle. And the best score on the kaggle comes from the full model.

2. preprocessing.py </li>This contains the preprocessing functions : standardization, replace_invalid_value

3. feature_engineering.py </li>It contains the operations we do to select or extend the features, and a function with the same name that combines all the operations together, by taking an class "ops" as an input, to tell this function to execute which operations in which order.

4. cross_validation.py </li>It contains the function to split the data and the cross validation functions for both ridge regression and logistic regression, returning the mean value of the scores.

5. Other basic files: </li>implementations.py, cost_functions, gradients.py, project_func.py, proj1_helpers.py
## Contributers
Tianchu ZHANG, Shiyuan HAO, Yishi JIA


[leaderboard]: https://www.kaggle.com/c/epfml18-higgs/leaderboard
[kaggle]: https://www.kaggle.com/c/epfml18-higgs


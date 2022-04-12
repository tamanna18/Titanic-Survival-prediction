import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve


def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    train_data.info()
    print("-" * 40)
    test_data.info()
    return train_data, test_data


def add_miss_data(train_data):
    embarked_mode = train_data['Embarked'].dropna().mode().values
    print(embarked_mode)
   
    train_data['Embarked'][train_data['Embarked'].isnull()] = embarked_mode
   
    train_data['Cabin'] = train_data['Cabin'].fillna('U0')
   
    age_df = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
    age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
    X = age_df_notnull.values[:, 1:]
    Y = age_df_notnull.values[:, 0]
    # use RandomForestRegression to train data
    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X, Y)
    predictAges = RFR.predict(age_df_isnull.values[:, 1:])
    train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges

    return train_data


def get_combined_data():
    train_df_org = pd.read_csv('data/train.csv')
    test_df_org = pd.read_csv('data/test.csv')
    test_df_org['Survived'] = 0
    combined_train_test = train_df_org.append(test_df_org)
    return combined_train_test


def process_embarked(combined_train_test):
    '''
    '''
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
    # df['column'] is a Series, but df[['column']] is a DataFrame
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],
                                    prefix=combined_train_test[['Embarked']].columns[0])
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

    return combined_train_test


def process_sex(combined_train_test):
    '''
    '''
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    # df['column'] is a Series, but df[['column']] is a DataFrame
    sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
    combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

    return combined_train_test


def process_name(combined_train_test):
    combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*)\.").findall(x)[0])
    # map similar titile to one specified
    title_dict = {}
    title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].map(title_dict)
    combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
    title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
    combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
    combined_train_test['Name_len'] = combined_train_test['Name'].map(len)

    return combined_train_test


def process_fare(combined_train_test):
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
        combined_train_test.groupby('Pclass').transform(np.mean))
    combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(combined_train_test['Ticket']).transform(
        'count')
    combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
    
    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
    combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
    fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
    combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
    combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)
    return combined_train_test


def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small'
    else:
        return 'Large'


def process_family_size(combined_train_test):
    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
    combined_train_test['Family_Cate'] = combined_train_test['Family_Size'].map(family_size_category)
    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small', 'Large']))
    combined_train_test['Family_Cate'] = le_family.transform(combined_train_test['Family_Cate'])
    family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Cate'],
                                            prefix=combined_train_test[['Family_Cate']].columns[0])
    combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)

    return combined_train_test


def process_age(combined_train_test):
    missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_len', 'Family_Size',
                                                       'Family_Cate', 'Fare', 'Fare_bin_id', 'Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
    missing_age_train_X = missing_age_train.drop(['Age'], axis=1)
    missing_age_train_Y = missing_age_train['Age']
    missing_age_test_X = missing_age_test.drop(['Age'], axis=1)
    # model gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_train_X, missing_age_train_Y)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train error for age:' + str(gbm_reg_grid.score(missing_age_train_X, missing_age_train_Y)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_test_X)
    print(missing_age_test['Age_GB'][:4])
    # model rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                               scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_train_X, missing_age_train_Y)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('GB Train error for age:' + str(rf_reg_grid.score(missing_age_train_X, missing_age_train_Y)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_test_X)
    print(missing_age_test['Age_RF'][:4])

    missing_age_test.loc[:, 'Age'] = missing_age_test[['Age_GB', 'Age_RF']].mean(axis=1)
    print(missing_age_test.loc[:, 'Age'])
    combined_train_test.loc[combined_train_test['Age'].isnull(), 'Age'] = missing_age_test['Age']

    return combined_train_test


def process_ticket(combined_train_test):
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(
        lambda x: 'U0' if x.isnumeric() else x)
    # combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    # combined_train_test['Ticket_Number'].fillna(0, inplace=True)
    #  Ticket_Letter factorize
    combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

    return combined_train_test


def process_cabin(combined_train_test):
    combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

    return combined_train_test


# PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'


def process_pclass(combined_train_test):
    Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
    Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
    Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]
    # Pclass_Fare Category
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(
        Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
    pclass_level = LabelEncoder()
    
    pclass_level.fit(np.array(
        ['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
    
    combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'
                                                                         ])
    # dummy 
    pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(
        columns=lambda x: 'Pclass' + str(x))
    combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)

    return  combined_train_test

def processs_correlation(combined_train_test):
    correlation = pd.DataFrame(combined_train_test[
                                   ['Embarked', 'Sex', 'Title', 'Name_len', 'Family_Size', 'Family_Cate', 'Fare',
                                    'Fare_bin_id', 'Pclass', 'Age', 'Ticket_Letter', 'Cabin']])
    colormap = plt.cm.viridis
    plt.figure(figsize=(14, 12))
    plt.title_dict('Person correlation of features', y=1.05, size=15)
    sns.heatmap(correlation.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
                linecolor='white', annot=True)


def regularization(combined_train_test):
    scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare', 'Name_len']])
    combined_train_test[['Age', 'Fare', 'Name_len']] = scale_age_fare.transform(
        combined_train_test[['Age', 'Fare', 'Name_len']])

    return combined_train_test


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(rf_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(rf_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X.columns),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # ada boost
    ada_est = ensemble.AdaBoostClassifier(random_state=42)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X.columns),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ensemble.ExtraTreesClassifier(random_state=42)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Features from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # merge the three models, and drop duplicates
    features_top_n = pd.concat([features_top_n_ada, features_top_n_et, features_top_n_ada, features_top_n_gb],
                               ignore_index=True).drop_duplicates()
    features_importance = pd.concat(
        [feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, feature_imp_sorted_gb,
         feature_imp_sorted_gb], ignore_index=True)

    return features_top_n, features_importance


def get_out_fold(clf, x_train, y_train, x_test):
    n_train = titanic_train_data_X.shape[0]
    n_test = titanic_test_data_X.shape[0]
    SEED = 0
    NFOLDS = 7
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)

    oof_train = np.zeros((n_train,))
    oof_test = np.zeros((n_test,))
    oof_test_skf = np.empty((NFOLDS, n_test))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def stacking_level_one(x_train, y_train, x_test):
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2,
                                    max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors=2)
    svm = SVC(kernel='linear', C=0.025)

    # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test)
    # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test)
    # AdaBoost
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test)
    # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test)
    # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test)
    # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test)
    # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test)

    oof_train = (rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train)
    oof_test = (rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test)

    return oof_train, oof_test


def stacking_level_two(x_train, y_train, x_test):
    gbm = XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1)
    gbm.fit(x_train, y_train)
    predictions = gbm.predict(x_test)
    print(predictions)
    return predictions


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        verbose=0):
    '''
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


def et_learn_and_output(x_train, y_train, x_test, PassengerId):
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    et.fit(x_train, y_train)
    predictions = et.predict(x_test)
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('ETSubmission.csv', index=False, sep=',')


def rf_learn_and_output(x_train, y_train, x_test, PassengerId):
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('RFSubmission.csv', index=False, sep=',')


def gbm_learn_and_output(x_train, y_train, x_test, PassengerId):
    gbm_parameters = {'n_estimators': 50, 'max_depth': 5, 'min_child_weight': 2, 'gamma': 0.9, 'subsample': 0.8,
                      'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'nthread': -1, 'scale_pos_weight': 1}
    gbm = XGBClassifier(**gbm_parameters)
    gbm.fit(x_train, y_train)
    predictions = gbm.predict(x_test)
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('GBMSubmission.csv', index=False, sep=',')


def show_learning_curves(x_train, y_train):
    X = x_train
    Y = y_train
    # RandomForest
    rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 5, 'min_samples_leaf': 2,
                     'max_features': 'sqrt', 'verbose': 0}
    # AdaBoost
    ada_parameters = {'n_estimators': 500, 'learning_rate': 0.1}
    # ExtraTrees
    et_parameters = {'n_jobs': -1, 'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}
    # GradientBoosting
    gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}
    # DecisionTree
    dt_parameters = {'max_depth': 8}
    # KNeighbors
    knn_parameters = {'n_neighbors': 2}
    # SVM
    svm_parameters = {'kernel': 'linear', 'C': 0.025}
    # XGB
    gbm_parameters = {'n_estimators': 50, 'max_depth': 5, 'min_child_weight': 2, 'gamma': 0.9, 'subsample': 0.8,
                      'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'nthread': -1, 'scale_pos_weight': 1}
    title = "Learning Curves"
    plot_learning_curve(XGBClassifier(**gbm_parameters), title, X, Y, cv=None, n_jobs=4,
                        train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500,550])
    plt.show()


if __name__ == '__main__':
    train_data_org, test_data_org = load_data()
    # add_miss_data(train_data)
    combined_train_test = get_combined_data()
    combined_train_test = process_embarked(combined_train_test)
    combined_train_test = process_sex(combined_train_test)
    combined_train_test = process_name(combined_train_test)
    combined_train_test = process_fare(combined_train_test)
    combined_train_test = process_family_size(combined_train_test)
    combined_train_test = process_age(combined_train_test)
    combined_train_test = process_ticket(combined_train_test)
    combined_train_test = process_cabin(combined_train_test)
    combined_train_test = process_pclass(combined_train_test)
    combined_train_test = regularization(combined_train_test)
    combined_train_test.drop(
        ['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Parch', 'SibSp', 'Family_Cate', 'Ticket','Pclass_Fare_Category'],
        axis=1, inplace=True)
    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]
    titanic_train_data_X = train_data.drop(['Survived'], axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'], axis=1)
    features_top_n, features_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, 20)
    # just use the selected features
    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[features_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[features_top_n])
    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    x_train = titanic_train_data_X.values
    # Creates an array of the train data
    y_train = titanic_train_data_Y.values
    # Creats an array of the test data
    x_test = titanic_test_data_X.values
    print(x_train.shape)
    print(combined_train_test.info())
    '''
    oof_train, oof_test = stacking_level_one(x_train,y_train,x_test)
    x_train_xg = np.concatenate(oof_train, axis=1)
    x_test_xg = np.concatenate(oof_test, axis=1)
    predictions = stacking_level_two(x_train_xg, y_train, x_test_xg)
    PassengerId = test_data_org['PassengerId']
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')
    '''
    PassengerId = test_data_org['PassengerId']
    #show_learning_curves(x_train, y_train)
    #rf_learn_and_output(x_train, y_train, x_test, PassengerId)
    gbm_learn_and_output(x_train, y_train, x_test, PassengerId)

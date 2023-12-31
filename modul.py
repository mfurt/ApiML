import numpy as np
import pandas as pd
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import datetime
import pickle
import shutil

param_grid = {'n_estimators': range(100, 1000, 100),
              'max_depth': range(1, 11)}

# метрика
def wape(y_true: np.array, y_pred: np.array):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# доступ до бд или использование локальных файлов
def read_files():
    try:
        df_pr = pd.read_json(execute())
    except:
        df_pr = pd.read_csv('data/pr_df.csv')
    try:
        df_sales_train = pd.read_json(execute())
    except:
        df_sales_train = pd.read_csv('data/sales_df_train.csv')
    try:
        df_st = pd.read_json(execute())
    except:
        df_st = pd.read_csv('data/st_df.csv')
    try:
        df_hol = pd.read_json(execute())
    except:
        df_hol = pd.read_csv('data/holidays_covid_calendar.csv')
    return df_pr, df_sales_train, df_st, df_hol

# сбор данных из различных моделей бд
def make_one(df_pr, df_sales_train, df_st, df_hol):
    df = df_sales_train.merge(df_pr, left_on='pr_sku_id', right_on='pr_sku_id')
    df = df.merge(df_st, left_on='st_id', right_on='st_id')
    df['date'] = pd.to_datetime(df['date'])
    df_hol['date'] = pd.to_datetime(df_hol['date'], format='%d.%m.%Y')
    df = df.merge(df_hol, left_on='date', right_on='date')
    return df, df_hol

# удаление нежелательных файлов
def ez_drop(df):
    number_list = [_ for _ in df.columns if (df[_].dtype == 'int64' or df[_].dtype == 'float64')]
    df = df[df['pr_sales_in_units'] > 0]
    df = df.drop(df[(df['pr_sales_in_rub'] == 0) & (df['pr_sales_in_units'] != 0)].index, axis=0)
    return number_list, df

# удаление ненужных данных
def ez_game(df):
    df = df.sort_values('date')
    df['month'] = df['date'].apply(lambda x: x.month)
    drop_set = ['pr_promo_sales_in_units', 'pr_promo_sales_in_rub', 'calday', 'covid', 'date']
    df = df.drop(columns=drop_set)
    market_list = df['st_id'].unique()
    city_list = df['st_city_id'].unique()
    return df, market_list, city_list

# генерация данных для обучения по магазинам
def generate_market_data(df, market_list):
    try:
        for _ in market_list:
            new_df = df.copy()
            new_df = new_df[new_df['st_id'] == _]
            market = new_df['st_id'].iloc[0]
            try:
                path = new_df['st_city_id'].iloc[0]
                os.makedirs(f"data/market_data/{path}")
            except FileExistsError:
                pass
            new_df = new_df.drop(['st_id', 'st_city_id'], axis=1)
            new_df.to_csv(f"data/market_data/{path}/market_data_{market}.csv")
    
    except:
        return 'Есть проблема'
    return new_df

# некоторые константы
def mini_const(new_df, df_sales_train):
    path = 'data/market_data'
    dir_list = os.listdir(path)
    max_date = df_sales_train['date'].astype('datetime64[ns]').max()
    str_list = [_ for _ in new_df.columns if new_df[_].dtype == 'object']

    return path, dir_list, max_date, str_list

# обучение моделей, с последющем сохранинем моделей для дальнейшего использования
def learn_cat(dir_list, path, param_grid, str_list):
    for i in dir_list:
        file_name = os.listdir(path+'/'+i)
        for j in file_name:
            df = pd.read_csv(f'data/market_data/{i}/{j}', index_col=0)
            y = df['pr_sales_in_units']
            X = df.drop(['pr_sales_in_units', 'pr_sales_in_rub'], axis=1)
            tsscv = TimeSeriesSplit()
            estimator = CatBoostRegressor(random_state=0, verbose = False)
            model_cbr = RandomizedSearchCV(estimator, 
                                      param_grid,
                                      verbose = False,
                                      n_jobs=-1, 
                                      cv=tsscv,
                                      scoring='neg_mean_squared_error',
                                      random_state=0)
            model_cbr.fit(X, y, cat_features=str_list)
            parametrs_cbr = model_cbr.best_params_
            best_model_cbr = model_cbr.best_estimator_
            try:
                os.makedirs(f"data/model/{i}")
            except FileExistsError:
                pass
            with open(f'data/model/{i}/model_{j[11:-4]}.pkl','wb') as f:
                pickle.dump(best_model_cbr,f)
    return X.columns

# генерация данных для предсказаний на 14 дней, с сохранением файлов
def generate_data_pred(dir_list, path, max_date, df, df_hol, feat_col):
    for i in dir_list:
        file_name = os.listdir(path + '/' + i)
        for j in file_name:
            new_date = max_date
            df_data_pred = pd.DataFrame(columns=feat_col)
            pr_df = pd.read_csv(f'data/market_data/{i}/{j}',
                                index_col=0).drop(['pr_sales_in_units', 'pr_sales_in_rub'], axis=1)
            for _ in range(14):
                new_date = new_date + datetime.timedelta(days=1)
                pred_df = pr_df.loc[pr_df['pr_sku_id'].drop_duplicates().index]
                pred_df['year'] = new_date.year
                pred_df['day'] = new_date.day
                pred_df['weekday'] =int(df_hol[df_hol['calday'] == int(new_date.strftime("%Y%m%d"))]['weekday'])
                pred_df['month'] = new_date.month
                pred_df['holiday'] = int(df_hol[df_hol['calday'] == int(new_date.strftime("%Y%m%d"))]['holiday'])
                df_data_pred = pd.concat([df_data_pred, pred_df])
                df_data_pred.reset_index(drop=True, inplace=True)
            try:
                os.makedirs(f"data/market_data_pred/{i}")
            except FileExistsError:
                pass
            df_data_pred.to_csv(f"data/market_data_pred/{i}/{j[:-4]}_pred.csv")

# получение предсказаний
def just_result(dir_list, path):
    for i in dir_list:
        file_name = os.listdir(path + '_pred/' + i)
        for j in file_name:
            result = pd.DataFrame(columns=['st_id', 'pr_sku_id', 'date', 'target'])
            pr_df = pd.read_csv(f'data/market_data_pred/{i}/{j}',
                                index_col=0)
            with open(f'data/model/{i}/model__{j[12:-9]}.pkl', 'rb') as f:
                model = pickle.load(f)

                pred = model.predict(pr_df)
            apply_all = np.vectorize(round)
            pred = apply_all(pred)

            result['st_id'] = [i for _ in range(len(pr_df))]
            result['pr_sku_id'] = pr_df['pr_sku_id']
            result['date'] = pd.to_datetime(dict(year=pr_df.year, month=pr_df.month, day=pr_df.day))
            result['target'] = pred

            try:
                os.makedirs(f"data/result/{i}")
            except FileExistsError:
                pass
            result.to_csv(f"data/result/{i}/{j[12:-9]}.csv")

# сохранение результатов в требуемый формат
def save_result(dir_list):
    sales = pd.DataFrame(columns=['st_id', 'pr_sku_id', 'date', 'target'])

    for i in dir_list:
        file_name = os.listdir('data/result/' + i)
        for j in file_name:
            pr_df = pd.read_csv(f'data/result/{i}/{j}',
                                index_col=0)
            sales = pd.concat([sales, pr_df])
    sales.to_csv('data/sales_submission.csv')

# удаление файлов
def drop_cash():
    shutil.rmtree("data/market_data_pred")
    shutil.rmtree("data/market_data")
    shutil.rmtree("data/result")
    shutil.rmtree("data/model")

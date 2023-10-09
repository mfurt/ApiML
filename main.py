from typing import Union
import uvicorn
from fastapi import FastAPI
from modul import *
from fastapi_utilities import repeat_at

app = FastAPI()


@app.get("/")
async def learn():
    df_pr, df_sales_train, df_st, df_hol = read_files()
    df, df_hol = make_one(df_pr, df_sales_train, df_st, df_hol)
    number_list, df = ez_drop(df)
    df, market_list, city_list = ez_game(df)
    new_df = generate_market_data(df, market_list)
    path, dir_list, max_date, str_list = mini_const(new_df, df_sales_train)
    learn_cat(dir_list, path, param_grid, str_list)
    generate_data_pred(dir_list, path, max_date, df, df_hol)
    just_result(dir_list, path)
    drop_cash()
    return "Успех"


@app.post('/yea')
def post_table():
    pass


if __name__ == "__main__":
    uvicorn.run(app, port=5000)

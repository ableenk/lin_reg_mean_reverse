'''Here you can get some parameters staff. Randomizing, some math, right choices, very good things, don't miss it.
'''

import json
import logging

import optuna

from gym import *
from backtests import *
from data_handler import *
from tools import *

fileh = logging.FileHandler('./opt_params_python.log', 'a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fileh.setFormatter(formatter)
log = logging.getLogger()
for hdlr in log.handlers[:]:
    log.removeHandler(hdlr)
log.addHandler(fileh)
log.setLevel(20)

def run_opt(trial, ask_prices, bid_prices, mid_prices, max_pos_usd, commission):
    '''Directly optimization function. Nice to meet you.
    '''
    k_assets = 30
    stop_loss_coef = trial.suggest_float("stop_loss_coef", 2, 50, step=0.1)
    std_coef = trial.suggest_float("std_coef", 0.4, 1.5, step=0.05)
    profit_coef = trial.suggest_float("profit_coef", 0, 1.2, step=0.05)
    # train_size = trial.suggest_int("train_size", 80000, 300000, step=10000)
    # ratio = trial.suggest_int("ratio", 1, 4)
    # test_size = train_size//ratio
    # ratio = trial.suggest_int("ratio", 2, 6)
    # test_size = train_size//ratio
    train_size = 70000
    test_size = 35000
    alpha = trial.suggest_float("alpha", 0, 2, step=0.01)
    dependent_asset = -1

    weights, means, uppers, lowers = get_data_for_backtest(mid_prices, train_size, test_size, max_pos_usd, std_coef, alpha=alpha, 
                                dependent_asset=dependent_asset)
    
    balance, UPLs, transactions_count = run_backtest(ask_prices[train_size:], bid_prices[train_size:], mid_prices[train_size:], 
                                weights, means, uppers, lowers, stop_loss_coef, profit_coef, commission, max_pos_usd)
    period = (0, weights.shape[0])
    profit_to_drawdown = 0
    wealth = (balance+UPLs)[period[0]:period[1]]
    if ((period[1] - period[0])//86000) * 5 > transactions_count:
        score = -10
    else:
        score = sharp_coef(wealth)
        if score > 0:
            score, profit_to_drawdown = get_score(wealth)
            if ((period[1] - period[0])//86000) * 20 > transactions_count:
                score /= 2
    res_dict = {'k_assets': k_assets,
                'stop_loss_coef': stop_loss_coef,
                'std_coef': std_coef,
                'profit_coef': profit_coef,
                'train_size': int(train_size),
                'test_size': int(test_size),
                'alpha': alpha,
                'transactions_count': transactions_count,
                'dependent_asset': int(dependent_asset), 
                'value': wealth[-1],
                'score': score,
                'profit_to_drawdown': profit_to_drawdown,
                'split_ts': train_size+period[1],
                'max_position_usd': max_pos_usd}
                
    json_res = json.dumps(res_dict)
    log.info(json_res)
    return profit_to_drawdown

def optimize_model():
    split = 1400000
    k_assets = 30
    ask = get_data(file='asks', end_ts=split)
    bid = get_data(file='bids', end_ts=split)
    mid = (ask+bid)/2
    best = selectksimilar(mid, k_assets)
    ask = ask[best]
    bid = bid[best]
    mid = mid[best]

    ask = ask.values
    bid = bid.values
    mid = mid.values

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: run_opt(trial, ask, bid, mid, 10000, 0.000153), n_trials=500, n_jobs=20)

if __name__ == '__main__':
    optimize_model()
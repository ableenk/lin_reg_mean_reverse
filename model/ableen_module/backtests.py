'''Different variations of backtests.
'''
import numpy as np
from numba import njit

from gym import *
from tools import *

    
# БЭКТЕСТ

@njit
def run_backtest(
    ask_prices: np.ndarray,
    bid_prices: np.ndarray,
    mid_prices: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    uppers: np.ndarray,
    lowers: np.ndarray,
    stop_loss_coef: float,
    profit_coef: float,
    commission: float, 
    max_position_usd: float,
    era_size=None,
    log=False
):
    '''Best backtest I have, I can't say I really love it, but it is good so far.
    '''
    out_of_gate = 0
    current_era=0
    transactions_count = 0
    current_action = 0
    assets_count = ask_prices.shape[1]

    start = (np.abs(weights).sum(axis=1)!=0).argmax(axis=0)
    trading_period = weights.shape[0]

    balances = np.zeros(trading_period)
    upl_array = np.zeros(trading_period)
    
    lc = linear_comb(mid_prices, weights)
    
    if start == -1:
        return balances, upl_array, transactions_count

    for ts in range(start+1, trading_period):
        balances[ts] = balances[ts-1]
        era_changed = uppers[ts] != uppers[ts-1]
        if era_changed:
            current_era += 1
        usd_distribution_array = weights[ts] * max_position_usd / np.abs(weights[ts]).sum()
        if not current_action:
            new_action = get_action(lc, means, uppers, lowers, ts)
            
        if (not current_action) and new_action and (new_action*out_of_gate != -1) and (uppers[ts] != 0):
            out_of_gate = 0
            current_action = new_action
            
            cur_positions = np.zeros(assets_count)
            for asset in range(assets_count):
                if usd_distribution_array[asset]*current_action > 0:
                    cur_positions[asset] = usd_distribution_array[asset] / ask_prices[ts][asset]
                else:
                    cur_positions[asset] = usd_distribution_array[asset] / bid_prices[ts][asset]

            balances[ts] += -current_action*usd_distribution_array.sum() - commission*max_position_usd

        elif current_action and pos_should_be_closed(current_action, lc, means, uppers, lowers, ts, stop_loss_coef, profit_coef):
            if era_changed:
                out_of_gate = int(lc[ts] > means[ts])*2-1
            else:
                transactions_count += 1
                out_of_gate = out_of_bounds(lc, means, uppers, lowers, ts, stop_loss_coef)
            balance_change = 0
            volume = 0
            for asset in range(assets_count):
                if cur_positions[asset] < 0:
                    change = cur_positions[asset] * ask_prices[ts][asset] * current_action
                else:
                    change = cur_positions[asset] * bid_prices[ts][asset] * current_action
                balance_change += change
                volume += np.abs(change)
            balances[ts] += balance_change - commission*volume
            current_action = 0

        if current_action != 0:
            new_upl = 0
            for asset in range(assets_count):
                if cur_positions[asset] < 0:
                    new_upl += cur_positions[asset] * ask_prices[ts][asset] * current_action
                else:
                    new_upl += cur_positions[asset] * bid_prices[ts][asset] * current_action
            upl_array[ts] = new_upl
    
    return balances, upl_array, transactions_count

def get_data_for_backtest(data_array, train_size, test_size, max_position_usd, std_coef=1, dependent_asset=7, alpha=0, need=None, min_correlation=None, low_threshold=0, corr_size=None):
    '''Preparing all data for backtest.
    Returns weights, mean values and thresholds in any time.
    '''
    if corr_size is None:
        corr_size = train_size

    eras_number = (data_array.shape[0]-train_size)//test_size
    size = eras_number * test_size + train_size

    all_weights = np.zeros((size, data_array.shape[1]))
    all_means = np.zeros(size)
    upper_thresholds = np.zeros(size)
    lower_thresholds = np.zeros(size)
    start = 0
    split = train_size
    end = train_size + test_size
    for era in range(eras_number):
        train_sample = data_array[start:split]
        correlated = np.ones(data_array.shape[1]).astype('bool')
        if correlated.sum() < 4:
            weights = np.zeros(test_size)
            mean_lc = 0
            upper_threshold = 0
            lower_threshold = 0
        else:
            weights, mean_lc, upper_threshold, lower_threshold = one_step_training(train_sample[:, correlated], test_size, std_coef, max_position_usd, dependent_asset, alpha=alpha)
            count = 0
            need = (correlated != 0).sum()
            for i in range(correlated.shape[0]):
                if count < need:
                    if correlated[i]:
                        all_weights[split:end, i] = weights[:, count]
                        count += 1

        all_means[split:end] = np.zeros(test_size) + mean_lc
        upper_thresholds[split:end] = np.zeros(test_size) + upper_threshold
        lower_thresholds[split:end] = np.zeros(test_size) + lower_threshold

        start += test_size
        split += test_size
        end += test_size

    return all_weights[train_size:], all_means[train_size:], upper_thresholds[train_size:], lower_thresholds[train_size:]
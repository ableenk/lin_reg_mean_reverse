import numpy as np

from numba import njit

@njit
def linear_comb(assets: np.ndarray, weights: np.ndarray):
    '''Returns linear combination of assets with related weights. 
    '''
    if weights.ndim != 1:
        if weights.shape[0] > assets.shape[0]:
            weights = weights[:assets.shape[0]]
        elif weights.shape[0] < assets.shape[0]:
            assets = assets[:weights.shape[0]]
    result = weights * assets
    return result.sum(axis=1)

def sharp_coef(wealthy: np.ndarray):
    '''Calculates sharp coefficient by returns.
    '''
    returns = wealthy[1:] - wealthy[:-1]
    if np.std(returns) == 0:
        return 0
    return returns.shape[0]**0.5 * np.mean(returns)/np.std(returns)

def get_score(wealthy: np.ndarray):
    _, _, max_drawdown = calculate_drawdown(wealthy)
    max_value = wealthy[-1]-wealthy[0]
    if max_value < 0:
        return sharp_coef(wealthy), 0
    parts_count = 6
    arrays = np.array_split(wealthy, parts_count)
    res = 0
    for i in range(parts_count):
        res += sharp_coef(arrays[i])
    
    return res * max_value / (parts_count * max_drawdown), max_value / max_drawdown

def score_zero_crossing(array, learn_freq):
    '''Count how many times linear combination had crossed mean value.
    '''
    array[::learn_freq] = 0
    signs_multiply = array[1:] * array[:-1]
    return (signs_multiply < 0).sum()

@njit
def out_of_bounds(lcs, means, uppers, lowers, ts, stop_loss_coef):
    '''Test for big deviation, tells which threshold have been exceeded
    by a linear combination.
    '''
    stop_upper = uppers[ts] + (uppers[ts]-means[ts])*stop_loss_coef
    stop_lower = lowers[ts] + (lowers[ts]-means[ts])*stop_loss_coef
    if lcs[ts] > stop_upper:
        return 1
    if lcs[ts] < stop_lower:
        return -1
    return 0

@njit
def pos_should_be_closed(current_action, lcs, means, uppers, lowers, ts, stop_loss_coef, profit_coef):
    '''Signal to close the position if needed.
    '''
    if current_action == -1:
        mean_crossed = lcs[ts] < means[ts] + (lowers[ts]-means[ts]) * profit_coef
    elif current_action == 1:
        mean_crossed = lcs[ts] > means[ts] + (uppers[ts]-means[ts]) * profit_coef
    era_changed = uppers[ts] != uppers[ts-1]
    out_of_b = out_of_bounds(lcs, means, uppers, lowers, ts, stop_loss_coef)
    return mean_crossed or era_changed or out_of_b

@njit
def get_action(lcs, means, uppers, lowers, ts):
    '''Returns which action should we do at the moment.
    '''
    if lcs[ts] > uppers[ts]:
        return -1
    if lcs[ts] < lowers[ts]:
        return 1
    return 0

def calculate_drawdown(pnl):
    '''Calculate drawdowns max value and durations.
    '''
    last_new_high_ts = 0
    current_max_equity = pnl[0]
    max_drawdown = 0
    max_drawdown_duration = 0
    
    durations_list = np.zeros(len(pnl))
    for ts in range(len(pnl)):
        if pnl[ts] >= current_max_equity:
            current_max_equity = pnl[ts]
            cur_duration = ts - last_new_high_ts
            if cur_duration > max_drawdown_duration:
                max_drawdown_duration = cur_duration
            last_new_high_ts = ts
        else:
            max_drawdown = max(max_drawdown, current_max_equity - pnl[ts])
        
        durations_list[ts] = ts - last_new_high_ts

    if pnl[-1] < current_max_equity:
        max_drawdown_duration = max(max_drawdown_duration, len(pnl) - last_new_high_ts)

    durations_list = durations_list[durations_list != 0]
    mean_duration = np.mean(durations_list)
    return max_drawdown_duration, mean_duration, max_drawdown

def count_ending_loss(lc, learn_freq):
    res = 0
    count = lc.shape[0]//learn_freq
    for i in range(1, count):
        res += np.abs(lc[(learn_freq)*(i+1)-1])
    return res
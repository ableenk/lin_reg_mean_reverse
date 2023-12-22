import matplotlib.pyplot as plt
import seaborn as sns

from tools import *
from backtests import *

class ModelVisualizing():
    '''Class for visualizing and easy getting data by params.
    '''

    def __init__(self, ask, bid):
        sns.set_theme(style='darkgrid')
        sns.color_palette('rainbow')
        self.ask = ask
        self.bid = bid
        self.mid = (ask+bid)/2

    def set_params(self, stop_loss_coef=7, std_coef=0.7, profit_coef=0.3, train_size=210000, test_size=210000, max_position_usd=1000,
                                dependent_asset=7, alpha=0, need=None, split_ts=None):
        '''Setting parameters for simulation.
        '''
        self.stop_loss_coef = stop_loss_coef
        self.std_coef = std_coef
        self.profit_coef = profit_coef
        self.train_size = train_size
        self.test_size = test_size
        self.dependent_asset = dependent_asset
        self.alpha = alpha
        self.need = need
        self.split = split_ts
        self.max_position_usd = max_position_usd
    
    def show_the_case(self, log=False):
        '''The easiest way to show full picture of how your new parameters work.
        '''
        weights, means, uppers, lowers = self.backtest_data(mid_prices=self.mid)
        end = min(self.split+86400*20, self.ask.shape[0])
        balance, UPLs, count = run_backtest(self.ask[self.train_size:end], self.bid[self.train_size:end], 
                                            self.mid[self.train_size:], weights, means, uppers, lowers, self.stop_loss_coef, self.profit_coef, 0.000153, self.max_position_usd, log=log, era_size=self.test_size)
        period = (self.train_size, self.mid.shape[0]-self.train_size)
        wealthy = (balance+UPLs)[:period[1]]
        lc = linear_comb(self.mid[self.train_size:self.train_size+weights.shape[0]], weights)

        fig = plt.figure(figsize=(18, 9))
        fig.patch.set_facecolor('white')
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(wealthy.shape[0]), wealthy)
        plt.subplot(1, 2, 2)
        plt.plot(uppers - means, color='r')
        plt.plot(lowers - means, color='r')
        plt.axvline(self.split-self.train_size, color='r')
        plt.plot(lc - means)
        if log:
            print('-'*100)
            print(f'Number of transactions is {count} with average {count/((period[1]-period[0])//86400)} a day.')
        return wealthy, count
    
    def backtest_data(self, mid_prices=None):
        if mid_prices is None:
            mid_prices = self.mid
        end = min(self.split+86400*20, mid_prices.shape[0])
        weights, means, uppers, lowers = get_data_for_backtest(mid_prices[:end], self.train_size, 
                                                               self.test_size, self.max_position_usd, self.std_coef, dependent_asset=self.dependent_asset, alpha=self.alpha, need=self.need)
        return weights, means, uppers, lowers
    
    def linear_combination(self, weights, mid_prices=None):
        if mid_prices is None:
            mid_prices = self.mid
        lc = linear_comb(mid_prices[self.train_size:], weights)
        return lc

    def get_scores(self):
        weights, means, uppers, lowers = self.backtest_data(mid_prices=self.mid)
        end = min(self.split+86400*20, self.ask.shape[0])
        balance, UPLs, count = run_backtest(self.ask[self.train_size:end], self.bid[self.train_size:end], self.mid[self.train_size:],
                                                        weights, means, uppers, lowers, self.stop_loss_coef, self.profit_coef, 0.000153, 1000, era_size=self.test_size)
        period = (self.train_size, self.mid.shape[0]-self.train_size)
        wealthy = (balance+UPLs)[:period[1]]
        split = self.split - self.train_size
        train_score = get_score(wealthy[:split])
        test_score = get_score(wealthy[split:])
        return train_score, test_score

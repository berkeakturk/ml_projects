import pandas as pd
import scipy.stats as stats

class DistributionAnalyzer():
    def __init__(self, df):
        self.stocks = sorted(df.columns)
        self.means = {}
        self.stds = {}
        
    def _calculate_dist_params(self, df):
        self.returns = df.pct_change().dropna()
        for stock in self.stocks:
            df_ = self.returns[stock]
            mean = df_.mean()
            std = df_.std()
            self.means[stock] = mean
            self.stds[stock] = std
        
        mean_df = pd.DataFrame.from_dict(self.means, orient='index', columns=['Mean'])
        std_df = pd.DataFrame.from_dict(self.stds, orient='index', columns=['Std'])
        self.param_df = pd.concat([mean_df, std_df], axis=1)

    def calculate_probabilities(self, df, uplift):
        self._calculate_dist_params(df)
        self.param_df['Z_score'] = (uplift - self.param_df['Mean']) / self.param_df['Std']
        self.param_df['Probability'] = 1 - stats.norm.cdf(self.param_df['Z_score'])
        return pd.DataFrame(self.param_df['Probability'])
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MediaBiasEDA:
    """Class to plot EDA of regional media bias"""
    def __init__(self, data_file, mbfc_file):
        self.data_file = data_file
        self.mbfc_file = mbfc_file
        self.df = None
        self.bias_hierarchy = {'left': -2, 'left-center': -1, 'neutral': 0, 'right-center': 1, 'right': 2}
        self.factual_mapping = {'low': 0, 'mixed': 1, 'high': 2}

    def load_data(self):
        """Load data and set up mappings"""
        self.df = pd.read_csv(self.data_file)
        mbfc = pd.read_csv(self.mbfc_file)
        self.df = self.df.merge(mbfc, left_on='base_url', right_on='source')
        self.df['year'] = pd.to_numeric(self.df['DATE'].astype(str).str[:4])

    def process_data(self):
        """Process and clean data"""
        self.df['bias_encoded'] = self.df['bias'].map(self.bias_hierarchy)
        self.df['factual_numeric'] = self.df['factual_reporting'].map(self.factual_mapping)
        self.df['country'] = self.df['country'].replace({
            'usa (44/180 press freedom)': 'united states',
            'usa (45/180 press freedom)': 'united states',
            'usa': 'united states',
            'guam (us territory)': 'united states',
            'united kingdom (scotland)': 'united kingdom',
            'united kingsom': 'united kingdom',
            'northern ireland (uk)': 'united kingdom',
            'italy (vatican city)': 'italy'
        })
        self.df['country'] = self.df['country'].str.title()

    def plot_average_tone_distribution(self):
        """Plot distribution of Average Tone"""
        ax = sns.histplot(data=self.df, x='Avg_Tone', bins=30, kde=True)
        ax.set_xlabel('Average Tone')
        ax.set_title('Average Tone Distribution')
        plt.show()

    def plot_bias_count(self):
        """Plot count of bias"""
        sns.countplot(data=self.df, x='bias')
        plt.title('Bias Distribution')
        plt.show()

    def plot_factual_by_country(self, top=True):
        """Plot average actuality score by country, top or bottom 10"""
        avg_factual = self.df.groupby('country')['factual_numeric'].mean().reset_index()
        avg_factual = avg_factual.sort_values('factual_numeric', ascending=not top).head(15)

        sns.barplot(data=avg_factual, x='factual_numeric', y='country', palette='BuGn_r_d' if top else 'BuGn_d')
        plt.xticks([0.00, 1.00, 2.00], ['Low', 'Mixed', 'High'])
        plt.xlabel("Average Factual Rating")
        plt.xlim(xmax=2.1)
        plt.ylabel("Country")
        plt.title(f"Average Factual Rating: {'Top' if top else 'Bottom'} 15")
        plt.show()

    def plot_bias_by_country(self, right=True):
        """Plor average bias by country, top 10 right or left countries"""
        avg_bias = self.df.groupby('country')['bias_encoded'].mean().reset_index()
        avg_bias = avg_bias.sort_values('bias_encoded', ascending=not right).head(10)

        sns.barplot(data=avg_bias, x='bias_encoded', y='country', palette='OrRd_r_d')
        plt.xticks([-2.00, -1.00, 0.00, 1.00, 2.00], ['Left', 'Left-Center', 'Neutral', 'Right-Center', 'Right'])
        plt.xlabel("Average Bias")
        plt.ylabel("Country")
        plt.title(f"Average Bias: {'Most Right Leaning' if right else 'Most Left Leaning'}")
        plt.show()

    def plot_tone_by_country(self, top=True):
        """Plot average tone by country, top or bottom 10"""
        avg_tone = self.df.groupby('country')['Avg_Tone'].mean().reset_index()
        avg_tone = avg_tone.sort_values('Avg_Tone', ascending=not top).head(10)

        sns.barplot(data=avg_tone, x='Avg_Tone', y='country', palette='PuBuGn_r_d' if top else 'PuBuGn_d')
        plt.xlabel("Average Tone")
        plt.xlim(xmin=-12, xmax=3)
        plt.ylabel("Country")
        plt.title(f"Average Tone: {'Top' if top else 'Bottom'} 10")
        plt.show()

#usage
eda = MediaBiasEDA('english_mbfc.csv', 'mbfc.csv')
eda.load_data()
eda.process_data()

eda.plot_average_tone_distribution()
eda.plot_bias_count()
eda.plot_factual_by_country(top=True)
eda.plot_factual_by_country(top=False)
eda.plot_bias_by_country(right=True)
eda.plot_bias_by_country(right=False)
eda.plot_tone_by_country(top=True)
eda.plot_tone_by_country(top=False)

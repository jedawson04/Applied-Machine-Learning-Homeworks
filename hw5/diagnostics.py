import pandas as pd
from plotnine import *
from pandas import Series
# runs some diagnostics to help us consider if we are achieving reasonable results

def main():
    seoul = pd.read_csv("seoulbike.csv")
    energy = pd.read_csv("energy.csv")
    penguins = pd.read_csv("penguins.csv")
    mnist = pd.read_csv("mnist1000.csv")
    print(Series.value_counts(penguins["label"]))
    # print("Seoulbike:", seoul["label"].describe())
    # print("Energy:", energy["label"].describe())
    # histogram = (
    #     ggplot(seoul)
    #         + aes(x="label")
    #         + geom_histogram(fill="white", color="black")
    #         + labs(title="Seoulbike Label Distribution", x="Number of Hourly Rentals", y="Count")
    # )
    # # display the chart to the user
    # ggsave(histogram, 'seoul', path="./figures/")
    
if __name__ == "__main__":
    main()
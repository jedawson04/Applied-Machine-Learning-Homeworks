import pandas as pd

# runs some diagnostics to help us consider if we are achieving reasonable results

def main():
    seoul = pd.read_csv("seoulbike.csv")
    energy = pd.read_csv("energy.csv")
    print("Seoulbike:", seoul["label"].describe())
    print("Energy:", energy["label"].describe())

if __name__ == "__main__":
    main()
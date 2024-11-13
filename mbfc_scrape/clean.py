import argparse
import pandas as pd


parser = argparse.ArgumentParser(prog="Convert raw mediabiasfactcheck.com dataset to a normalized dataset with political bias and factual reporting labels")
parser.add_argument("-i", "--input-file", help="Path to the mbfc_raw.csv file", default="mbfc_raw-1731489978.csv")
parser.add_argument("-o", "--output-file", help="Path to save the output dataset", default="mbfc.csv")
args = parser.parse_args()

VALID_POLITICAL_BIAS_LABELS = ["left", "left-center", "neutral", "right-center", "right"]


def get_norm_bias_label(label):
    if "least" in label or ("pro-" in label and "science" in label):
        return "neutral"
    elif "center" in label and "right" in label:
        return "right-center"
    elif "center" in label and "left" in label:
        return "left-center"
    elif "right" in label:
        return "right"
    elif "left" in label:
        return "left"
    return label


def get_norm_factual_label(label):
    if "high" in label:
        return "high"
    elif "low" in label:
        return "low"
    elif "mostly factual" or "mixed" in label:
        return "mixed"
    return label


df = pd.read_csv(args.input_file)
df.dropna(subset=["source", "bias", "factual_reporting"], inplace=True)

df.bias = df.bias.map(get_norm_bias_label)
df.loc[~df.bias.isin(VALID_POLITICAL_BIAS_LABELS), "bias"] = None
df.factual_reporting = df.factual_reporting.map(get_norm_factual_label)
df.dropna(subset=["source", "bias", "factual_reporting"], inplace=True)

print("STATS:")
print("Dataset size:", len(df))
print()
print("Political Bias Label distribution:")
print(df.bias.value_counts())
print()
print("Factual Reporting Label distribution:")
print(df.factual_reporting.value_counts())

df[["source","country","bias", "factual_reporting"]].to_csv(args.output_file, index=False)
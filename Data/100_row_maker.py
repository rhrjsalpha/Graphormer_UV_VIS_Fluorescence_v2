import pandas as pd

csv_path = "DB for chromophore_Sci_Data_rev02.csv"

df = pd.read_csv(csv_path)
print(df.shape)
new_df = df[0:100]
print(new_df)
print(new_df.shape)
new_csv_path = csv_path + "_100row"
new_df.to_csv(new_csv_path)
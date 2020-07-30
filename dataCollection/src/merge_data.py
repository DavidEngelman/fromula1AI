

import os
import pandas as pd

ds_name = "../logs/1595951599300652"

df_telemetry = pd.read_csv(f"{ds_name}/telemetry_data.csv", index_col=None)
df_g29 = pd.read_csv(f"{ds_name}/g29_data.csv", index_col=None)

precision = 3
c = 0

to_filter = [f[:-4 - precision] for f in os.listdir(f"{ds_name}/img/")]

df_telemetry = df_telemetry[df_telemetry["clock"].str.contains('|'.join(to_filter))]
df_g29 = df_g29[df_g29["clock"].str.contains('|'.join(to_filter))]

df_telemetry = df_telemetry.drop_duplicates(subset='clock', keep="last")
df_g29 = df_g29.drop_duplicates(subset='clock', keep="last")

print(len(df_g29), len(df_telemetry))

print(df_g29)
print("qqzdqzdqzd")

# print(df_g29)
# print(df_telemetry)

df_telemetry['inputs'] = ''
for val, input in zip(df_g29.clock, df_g29.inputs):
    df_telemetry.loc[df_telemetry['clock'].str.contains(val[:- precision]), 'inputs'] = input


df_all = df_telemetry[df_telemetry["inputs"] != ""]
print(len(df_all))

df_all.to_csv("test.csv")
'''mport pandas as pd

# Create a sample DataFrame
data = {'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Value': [10, 20, 30, 40, 50, 60]}
df = pd.DataFrame(data)

# Group by Category and calculate percentage
df['Percentage'] = df.groupby('Category')['Value'].transform(lambda x: x / x.sum())

print(df)



Output:



  Category  Value  Percentage
0        A     10       0.333
1        A     20       0.667
2        B     30       0.429
3        B     40       0.571
4        C     50       0.455
5        C     60       0.545
'''

df.groupby( # type: ignore
 [
 "ord_1",
"ord_2"
]
)["id"].count().reset_index(name="count")

df.groupby(["ord_2"])["id"].transform("count") # type: ignore

'''One more trick is to create new features from these categorical variables. You can
create new categorical features from existing features, and this can be done in an
effortless manner.
═════════════════════════════════════════════════════════════════════════
In [X]: df["new_feature"] = (
...: df.ord_1.astype(str)
...: + "_"
...: + df.ord_2.astype(str)
...: )
In [X]: df.new_feature
Out[X]:
0 Contributor_Hot
1 Grandmaster_Warm
2 nan_Freezing
3 Novice_Lava Hot
4 Grandmaster_Cold
'''
# Use log transformation for high variance features than other columns

df.f_3.var()
df.f_3.apply(lambda x: np.log(1+x)).var()
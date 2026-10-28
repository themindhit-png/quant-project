import sys, pandas as pd
df = pd.read_csv(sys.argv[1]); df = df[df.p2 == 'pass']
d = pd.to_datetime(df.start) + pd.to_timedelta(df.total_days.round(), unit='D')
d = d[d <= '2026-07-26']; print(','.join(str(x.date()) for x in d))

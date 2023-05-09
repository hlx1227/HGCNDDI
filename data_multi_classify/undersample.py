import pandas as pd

df = pd.read_csv('graph_Moderate_replace.csv')
# moderate下采样
moderate_length = len(df['Level'].tolist())
moderate_delete_num = moderate_length - 30000
for i in range(0, moderate_delete_num):
    print(i)
    df = df.drop(i)

df.to_csv('../data_multi_classify_undersample/moderate_under_30000.csv', index=False)

# df2 = pd.read_csv('graph_Major_replace.csv')
# # major下采样
# major_length = len(df2['Level'].tolist())
# major_delete_num = major_length - 8629
# for i in range(0, major_delete_num):
#     print(i)
#     df2 = df2.drop(i)
#
# df2.to_csv('major_under_8629.csv', index=False)

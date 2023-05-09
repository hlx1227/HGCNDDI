import pandas as pd

# df = pd.read_csv('useSmileinKnownUnder.csv')
# # 1.获得replace
# replaced = pd.DataFrame()
# replaced['all'] = df['ID']
# replaced['replace'] = list(range(len(df['ID'].tolist())))
#
# replaced.to_csv('replace.csv', index=False)  # 不要索引
#
# # 2.进行replace,获得interaction
# replace = pd.read_csv('replace.csv')
# graph = pd.read_csv('graph_under.csv')
# drugbefore = replace['all'].tolist()
# drugafter = replace['replace'].tolist()
#
# drugA = graph['ID_A'].tolist()
# drugB = graph['ID_B'].tolist()
# Level = graph['Level'].tolist()
# finalA = []
# finalB = []
#
# # # 删除重复
# # druglist = list(set(drug))
# graph_re = pd.DataFrame()
# smiles = []
#
# num = 0
#
# for name in drugA:
#     for i, j in enumerate(drugbefore):
#         if name == j:
#             a = drugafter[i]
#             # num += 1
#             # print('多少了？', num)
#             finalA.append(drugafter[i])
#
# for name in drugB:
#     for i, j in enumerate(drugbefore):
#         if name == j:
#             b = drugafter[i]
#             finalB.append(drugafter[i])
#
# graph_re['ID_A'] = finalA
# graph_re['ID_B'] = finalB
# graph_re['Level'] = Level
# graph_re.to_csv('graph_replace_under.csv', index=False)  # 不要索引

# 3.划分Major,Minor,Moderate等
graph_replace = pd.read_csv('graph_replace_under.csv')
major = graph_replace[graph_replace['Level'] == 'Major']
major.to_csv('graph_Major_replace.csv', index=False)
minor = graph_replace[graph_replace['Level'] == 'Minor']
minor.to_csv('graph_Minor_replace.csv', index=False)
Moderate = graph_replace[graph_replace['Level'] == 'Moderate']
Moderate.to_csv('graph_Moderate_replace.csv', index=False)

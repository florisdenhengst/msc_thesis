import torch
beam_width = 3
c = torch.tensor(
         [[[3.2627, 2.5468, 2.4580],
         [3.3022, 2.5802, 2.5706],
         [3.3003, 2.6490, 2.5443],
         [3.2691, 2.5507, 2.4859],
         [3.2645, 2.4218, 2.3176],
         [3.3410, 2.4484, 2.3931],
         [3.2729, 2.4524, 2.4132],
         [3.1521, 2.3336, 2.2427],
         [3.2537, 2.3378, 2.3052],
         [3.2977, 2.4161, 2.3138],
         [3.4014, 2.6303, 2.5743],
         [3.2614, 2.5393, 2.4741],
         [3.2710, 2.4441, 2.3696],
         [3.2130, 2.3879, 2.3085],
         [3.3379, 2.4739, 2.4404],
         [3.2648, 2.3414, 2.3391],
         [3.3433, 2.5120, 2.4848],
         [3.3440, 2.3719, 2.3707],
         [3.1397, 2.5805, 2.4913],
         [3.2979, 2.3966, 2.3843],
         [3.2188, 2.5874, 2.5778],
         [3.2341, 2.5446, 2.5095],
         [3.1670, 2.4473, 2.3523],
         [3.2488, 2.2500, 2.2467],
         [3.1406, 2.3543, 2.3243]],
         [[3.2627, 2.5468, 2.4580],
         [3.3022, 2.5802, 2.5706],
         [3.3003, 2.6490, 2.5443],
         [3.2691, 2.5507, 2.4859],
         [3.2645, 2.4218, 2.3176],
         [3.3410, 2.4484, 2.3931],
         [3.2729, 2.4524, 2.4132],
         [3.1521, 2.3336, 2.2427],
         [3.2537, 2.3378, 2.3052],
         [3.2977, 2.4161, 2.3138],
         [3.4014, 2.6303, 2.5743],
         [3.2614, 2.5393, 2.4741],
         [3.2710, 2.4441, 2.3696],
         [3.2130, 2.3879, 2.3085],
         [3.3379, 2.4739, 2.4404],
         [3.2648, 2.3414, 2.3391],
         [3.3433, 2.5120, 2.4848],
         [3.3440, 2.3719, 2.3707],
         [3.1397, 2.5805, 2.4913],
         [3.2979, 2.3966, 2.3843],
         [3.2188, 2.5874, 2.5778],
         [3.2341, 2.5446, 2.5095],
         [3.1670, 2.4473, 2.3523],
         [3.2488, 2.2500, 2.2467],
         [3.1406, 2.3543, 2.3243]],
         [[3.2627, 2.5468, 2.4580],
         [3.3022, 2.5802, 2.5706],
         [3.3003, 2.6490, 2.5443],
         [3.2691, 2.5507, 2.4859],
         [3.2645, 2.4218, 2.3176],
         [3.3410, 2.4484, 2.3931],
         [3.2729, 2.4524, 2.4132],
         [3.1521, 2.3336, 2.2427],
         [3.2537, 2.3378, 2.3052],
         [3.2977, 2.4161, 2.3138],
         [3.4014, 2.6303, 2.5743],
         [3.2614, 2.5393, 2.4741],
         [3.2710, 2.4441, 2.3696],
         [3.2130, 2.3879, 2.3085],
         [3.3379, 2.4739, 2.4404],
         [3.2648, 2.3414, 2.3391],
         [3.3433, 2.5120, 2.4848],
         [3.3440, 2.3719, 2.3707],
         [3.1397, 2.5805, 2.4913],
         [3.2979, 2.3966, 2.3843],
         [3.2188, 2.5874, 2.5778],
         [3.2341, 2.5446, 2.5095],
         [3.1670, 2.4473, 2.3523],
         [3.2488, 2.2500, 2.2467],
         [3.1406, 2.3543, 2.3243]]])



t = torch.stack([torch.stack([c[i][j] for i in range(c.shape[0])]).flatten() for j in range(c.shape[1])])
# print(c.shape)
# print(t)
# print(t.shape)
idxs = torch.topk(t, k=beam_width, dim=1)[1]
x = [idx // beam_width for idx in idxs]
y = [idx % beam_width for idx in idxs]

xs = []
ys = []
for i, line_prob in enumerate(t):
   unique_probs, unique_ids = torch.unique(line_prob, return_inverse=True)
   # print(unique_probs)
   # print(unique_ids)
   topk_probs, topk_ids = torch.topk(unique_probs, k=beam_width)
   # topk_probs, topk_ids = torch.topk(line_prob, k=beam_width)
   # print(i)
   # print(topk_probs)
   # print(topk_ids)
   # print(line_prob[unique_ids[topk_ids]])
   xs.append([idx // beam_width for idx in unique_ids[topk_ids]])
   ys.append([idx % beam_width for idx in unique_ids[topk_ids]])
   # xs.append([idx // beam_width for idx in topk_ids])
   # ys.append([idx % beam_width for idx in topk_ids])
   # print(x)
   # print(y)
   
   print([c[xs[i][j]][i,ys[i][j]] for j in range(beam_width)])
   print([c[x[i][j]][i,y[i][j]] for j in range(beam_width)])


   # print(t[i])

# print(t.shape)
# print(t)
# i1, i2 = torch.unique(t, dim=1, return_inverse=True)
# print(i1)
# print(i2)
# print(torch.topk(t, k=beam_width, dim=1))
# print(torch.topk(torch.unique(t, dim=1), k=beam_width, dim=1))

print(xs)
print(ys)

print(x)
print(y)

# print([c[x[0][i]][0, y[0][i]] for i in range(beam_width)])

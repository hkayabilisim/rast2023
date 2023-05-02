#%%
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

aid_classes = ['airport', 'bareland', 'baseballfield', 'beach', 'bridge', 'center', 'church', 'commercial', 'denseresidential', 'desert', 'farmland', 'forest', 'industrial', 'meadow', 'mediumresidential', 'mountain', 'park', 'parking', 'playground', 'pond', 'port', 'railwaystation', 'resort', 'river', 'school', 'sparseresidential', 'square', 'stadium', 'storagetanks', 'viaduct']
nwpu_classes = ['airport', 'beach', 'bridge', 'church', 'desert', 'forest', 'meadow', 'mountain', 'river', 'stadium']

res = np.loadtxt('results/aid301_test.confmat')
vit = np.loadtxt('results/aid230_test.confmat')

true_decisions_res = 0 
true_decisions_vit = 0 
for row, nwpu_class in enumerate(nwpu_classes):
    col = aid_classes.index(nwpu_class)
    true_decisions_vit += vit[row,col]
    true_decisions_res += res[row,col]

acc_res = true_decisions_res / np.sum(res)
acc_vit = true_decisions_vit / np.sum(vit)

print('Accuracy of ResNet18           is: ', acc_res)
print('Accuracy of Vision Transformer is: ', acc_vit)


fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(6,4),sharex=True)
fig.set_tight_layout(True)
# change annotation font size
plt.rcParams["font.size"] = "6"
vit_hmap = vit.copy()
vit_hmap[vit_hmap == 0] = np.nan
res_hmap = res.copy()
res_hmap[res_hmap == 0] = np.nan
sns.heatmap(vit_hmap, annot=True, fmt='3.0f', cmap='Blues',ax=ax[0], cbar=False)
sns.heatmap(res_hmap, annot=True, fmt='3.0f', cmap='Blues',ax=ax[1], cbar=False)
ax[0].set_xticklabels(aid_classes, rotation=90)
ax[0].set_yticks(np.arange(len(nwpu_classes)))
ax[0].set_yticklabels(nwpu_classes, rotation=0)
ax[1].set_xticklabels(aid_classes, rotation=90)
ax[1].set_yticklabels(nwpu_classes, rotation=0)
ax[0].set_title('Vision Transformer (vit_tiny_r_s16_p8_224)')
ax[1].set_title('ResNet (resnet18)')
print('figs/confmat.pdf created!')
fig.savefig('figs/confmat.pdf')
# sns heatmap xticklabels with array of strings



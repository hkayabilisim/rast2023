#%%
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import json

def get_model_benchmark(m):
    with open('results/model_benchmarks.json') as f:
        benchmarks = json.load(f)
    for b in benchmarks:
        if b['model'] == m:
            return b
        
matplotlib.rcParams.update({'font.size': 6})
# others, convnextv2 and convnext
#ids = [29,30,31,32,33,34,35,36,37,38,39,40,41,     13,14,15,16,6,17,18,7,19,   20,21,22,23,24,25,9,26,27,28]

# there is a problem in the following jobs
# after a few epochs, we observe nan's.
# 18, 19, 28, 204, 206, 209, 210,212, 221, 310, 318, 319, 325,225
# swing errors : 102, 104,110,111,112,113,114,118,119,120,121,122,123,124 (mismatch in size)  
#       11,   # rex
#       10,   # convit
#       12,   # maxvit_rmlp_small_rw_224

# mobilenets 40 , 41 are eliminated very bad accuracy
# 31, 32, 34

# 319 323 sonradan ekle
vit_family = [201,202,203,205,207,208,213,214,215,217,218,219,220,222,223,224,226,227,228,229,230]
swin_family = [8,101,103,105,106,107,108,109,115,116,117]
resnet_family = [301,302,303,304,305,306,307,308,309,311,312,313,314,315,316,317,320,321,322,324]
convnext_family =   [13,14,15,16,6,17,7] + [20,21,22,23,24,25,9,26,27]
others = [29,30,33,35,36,37,38,39]


ids = swin_family + convnext_family + others + vit_family + resnet_family

def get_color(id):
    if id in swin_family:
        return 'tab:blue'
    elif id in convnext_family:
        return 'tab:orange'
    elif id in others:
        return 'tab:green'
    elif id in vit_family:
        return 'tab:red'
    elif id in resnet_family:
        return 'tab:purple'
    else:
        return 'black'

models = {4: 'xresnet50 nopre',
    5: 'convnextv2_tiny nopre',
    6: 'convnextv2_tiny',
    7: 'convnextv2_large',
    8: 'swin_large_patch4_window7_224',
    9: 'convnext_base',
    10: 'convit_base',
    11: 'rexnet_200',
    12: 'maxvit_rmlp_small_rw_224',
    13: 'convnextv2_atto',
    14: 'convnextv2_femto',
    15: 'convnextv2_pico',
    16: 'convnextv2_nano',
    17: 'convnextv2_small',
    18: 'convnextv2_base',
    19: 'convnextv2_huge',
    20: 'convnext_atto',
    21: 'convnext_femto',
    22: 'convnext_pico',
    23: 'convnext_nano',
    24: 'convnext_tiny',
    25: 'convnext_small',          
    26: 'convnext_large',
    27: 'convnext_xlarge',
    28: 'convnext_xxlarge',
    29: 'inception_resnet_v2',
    30: 'xception',
    31: 'tf_efficientnetv2_m',
    32: 'tf_efficientnet_b6',
    33: 'resnet50',
    34: 'densenet169',
    35: 'tf_inception_v3',
    36: 'vgg11',
    37: 'vgg13',
    38: 'vgg16',
    39: 'vgg19',
    40: 'mobilenetv3_small_100',
    41: 'mobilenetv3_large_100',
    101: 'swin_base_patch4_window7_224',
    102: 'swin_base_patch4_window12_384',
    103: 'swin_large_patch4_window7_224',
    104: 'swin_large_patch4_window12_384',
    105: 'swin_s3_base_224',
    106: 'swin_s3_small_224',
    107: 'swin_s3_tiny_224',
    108: 'swin_small_patch4_window7_224',
    109: 'swin_tiny_patch4_window7_224',
    110: 'swinv2_base_window8_256',
    111: 'swinv2_base_window12_192',
    112: 'swinv2_base_window12to16_192to256',
    113: 'swinv2_base_window12to24_192to384',
    114: 'swinv2_base_window16_256',
    115: 'swinv2_cr_small_224',
    116: 'swinv2_cr_small_ns_224',
    117: 'swinv2_cr_tiny_ns_224',
    118: 'swinv2_large_window12_192',
    119: 'swinv2_large_window12to16_192to256',
    120: 'swinv2_large_window12to24_192to384',
    121: 'swinv2_small_window8_256',
    122: 'swinv2_small_window16_256',
    123: 'swinv2_tiny_window8_256',
    124: 'swinv2_tiny_window16_256',
    201: 'vit_base_patch16_224',
    202: 'vit_base_patch16_224_miil',
    203: 'vit_base_patch16_clip_224',
    204: 'vit_base_patch16_rpn_224',
    205: 'vit_base_patch32_224',
    206: 'vit_base_patch32_clip_224',
    207: 'vit_base_patch8_224',
    208: 'vit_base_r50_s16_224',
    209: 'vit_giant_patch14_clip_224',
    210: 'vit_gigantic_patch14_clip_224',
    211: 'vit_huge_patch14_224',
    212: 'vit_huge_patch14_clip_224',
    213: 'vit_large_patch14_clip_224',
    214: 'vit_large_patch16_224',
    215: 'vit_large_patch32_224',
    216: 'vit_large_r50_s32_224',
    217: 'vit_relpos_base_patch16_224',
    218: 'vit_relpos_base_patch16_clsgap_224',
    219: 'vit_relpos_medium_patch16_224',
    220: 'vit_relpos_medium_patch16_cls_224',
    221: 'vit_relpos_medium_patch16_rpn_224',
    222: 'vit_relpos_small_patch16_224',
    223: 'vit_small_patch16_224',
    224: 'vit_small_patch32_224',
    225: 'vit_small_patch8_224',
    226: 'vit_small_r26_s32_224',
    227: 'vit_srelpos_medium_patch16_224',
    228: 'vit_srelpos_small_patch16_224',
    229: 'vit_tiny_patch16_224',
    230: 'vit_tiny_r_s16_p8_224',
    301: 'resnet18',
    302: 'resnet26',
    303: 'resnet32ts',
    304: 'resnet33ts',
    305: 'resnet34',
    306: 'resnet50',
    307: 'resnet51q',
    308: 'resnet61q',
    309: 'resnet101',
    310: 'resnet152',
    311: 'resnetaa50',
    312: 'resnetblur50',
    313: 'resnetrs50',
    314: 'resnetrs101',
    315: 'resnetrs152',
    316: 'resnetrs200',
    317: 'resnetrs270',
    318: 'resnetrs350',
    319: 'resnetrs420',
    320: 'resnetv2_50',
    321: 'resnetv2_101',
    322: 'resnetv2_101x1_bitm',
    323: 'resnetv2_101x3_bitm',
    324: 'resnetv2_152x2_bitm',
    325: 'resnetv2_152x4_bitm'
          }

df_all = {} 
for id in ids:
    df = pd.read_csv(f'results/aid{id}.csv',delim_whitespace=True)
    df['train_accuracy'] = 100*df['train_accuracy']
    df['valid_accuracy'] = 100*df['valid_accuracy']
    df['epoch'] = 1 + df['epoch']
    df[['minute','seconds']] = df['time'].str.split(':',expand=True)
    df = df.astype({'minute': 'int', 'seconds': 'int'})
    df['duration'] = df['minute']*60.0 + df['seconds']
    df_all[id] = df

for epoch in [299]:
    fig, ax = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(7,3.5))
    fig.set_tight_layout(True)
    for id in ids:
        #max_train_accuracy = df_all[id].describe()['train_accuracy']['max']
        #max_valid_accuracy = df_all[id].describe()['valid_accuracy']['max']
        #avg_duration_per_epoch =  df_all[id]['duration'].mean()

        valid_accuracy_at_epoch = df_all[id].iloc[epoch]['valid_accuracy']
        valid_error_at_epoch = 100 - valid_accuracy_at_epoch
        benchmark = get_model_benchmark(models[id])
        param_count = benchmark['param_count']
        train_step_time = benchmark['train_step_time']
        infer_step_time = benchmark['infer_step_time']
        infer_samples_per_sec = benchmark['infer_samples_per_sec']
        train_samples_per_sec = benchmark['train_samples_per_sec']

        x = train_step_time
        y = valid_error_at_epoch
        s = 3*benchmark['param_count']
        #print(f'exp{id:2d} {models[id]:30s} {x:.2f} {y:.2f} ')
        #label = label='_nolegend_'
        if id == 25:
            label = 'ConvNeXT'
        elif id == 29:
            label = 'Legacy'
        elif id == 106:
            label = 'Swin'
        elif id == 220:
            label = 'ViT'
        elif id == 307:
            label = 'ResNet'
        else:
            label = '_nolegend_'
        #if infer_step_time < 30 and y < 10:
        #    print(id)
        ax[0].scatter(train_step_time,y,s,label=label,
                   color=get_color(id),alpha=0.8,edgecolors='black')
        ax[1].scatter(infer_step_time,y,s,label=label,
                   color=get_color(id),alpha=0.8,edgecolors='black')
        #ax.text(x,y,f'{id}',fontdict={'size':6})
    ax[0].set_xlabel('training step time (miliseconds)',fontdict={'size':8})
    ax[1].set_xlabel('infererence step time (miliseconds)',fontdict={'size':8})
    ax[0].set_ylabel('validation error (%)',fontdict={'size':8})
    ax[1].set_ylabel('validation error (%)',fontdict={'size':8})
    #ax.set_yscale('log')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    #ax.set_yticks([4,5,6,7,8,9,10,20,30,40,50,60])
    #ax.set_yticklabels([4,5,6,7,8,9,10,20,30,40,50,60])
    #ax.set_ylim(4,60.0)    
    #ax.grid('both')
    ax[0].tick_params(labelsize=8)
    ax[1].tick_params(labelsize=8)
    ax[0].set_title('Training Performance vs Accuracy',fontdict={'size':8})
    ax[1].set_title('Inference Performance vs Accuracy',fontdict={'size':8})
    ax[0].legend()
    ax[1].legend()
    plt.savefig(f'figs/error_vs_time.pdf')


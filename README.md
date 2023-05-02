# Evaluation of Modern Deep Learning Architectures in Remote Sensing Scene Classification
This reposity is created to provide supplemantary information regarding the manuscript "Evaluation of Modern Deep Learning Architectures in Remote Sensing Scene Classification", H. Kaya and G. Taskin submitted to RAST 2023 conference.

## List of the models
This is the list of deep-learning models used in the manuscript. The first column corresponds to unique id specific to this study. Please note that, there are many other models in TIMM library, but we couldn't use all of them for practical issues such as dimension mismatches or memory requirements.

    201 vit_base_patch16_224
    202 vit_base_patch16_224_miil
    203 vit_base_patch16_clip_224
    205 vit_base_patch32_224
    207 vit_base_patch8_224
    208 vit_base_r50_s16_224
    213 vit_large_patch14_clip_224
    214 vit_large_patch16_224
    215 vit_large_patch32_224
    217 vit_relpos_base_patch16_224
    218 vit_relpos_base_patch16_clsgap_224
    219 vit_relpos_medium_patch16_224
    220 vit_relpos_medium_patch16_cls_224
    222 vit_relpos_small_patch16_224
    223 vit_small_patch16_224
    224 vit_small_patch32_224
    226 vit_small_r26_s32_224
    227 vit_srelpos_medium_patch16_224
    228 vit_srelpos_small_patch16_224
    229 vit_tiny_patch16_224
    230 vit_tiny_r_s16_p8_224
    8 swin_large_patch4_window7_224
    101 swin_base_patch4_window7_224
    103 swin_large_patch4_window7_224
    105 swin_s3_base_224
    106 swin_s3_small_224
    107 swin_s3_tiny_224
    108 swin_small_patch4_window7_224
    109 swin_tiny_patch4_window7_224
    115 swinv2_cr_small_224
    116 swinv2_cr_small_ns_224
    117 swinv2_cr_tiny_ns_224
    29 inception_resnet_v2
    30 xception
    33 resnet50
    35 tf_inception_v3
    36 vgg11
    37 vgg13
    38 vgg16
    39 vgg19
    13 convnextv2_atto
    14 convnextv2_femto
    15 convnextv2_pico
    16 convnextv2_nano
    6 convnextv2_tiny
    17 convnextv2_small
    7 convnextv2_large
    20 convnext_atto
    21 convnext_femto
    22 convnext_pico
    23 convnext_nano
    24 convnext_tiny
    25 convnext_small
    9 convnext_base
    26 convnext_large
    27 convnext_xlarge
    301 resnet18
    302 resnet26
    303 resnet32ts
    304 resnet33ts
    305 resnet34
    306 resnet50
    307 resnet51q
    308 resnet61q
    309 resnet101
    311 resnetaa50
    312 resnetblur50
    313 resnetrs50
    314 resnetrs101
    315 resnetrs152
    316 resnetrs200
    317 resnetrs270
    320 resnetv2_50
    321 resnetv2_101
    322 resnetv2_101x1_bitm
    324 resnetv2_152x2_bitm

## Template for Fine-tuning
We've used the following code template by replacing "MODEL_NAME" with the model names listed in the previous section. The code is run via "accelerate run" command. It can use CUDA as an accelerator without any change. Please note that you should prepare AID and RESISC45 folders in advance. It is possible to optimize the hyper-parameters a bit, but we tend to choose default options because it would be utterly difficult to optimize for each model.

```python
import timm
from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from accelerate.utils import write_basic_config
write_basic_config()

path = 'AID'
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=RandomSplitter(valid_pct=0.5, seed=0),
    get_items=get_image_files, get_y=parent_label,
    item_tfms=[RandomResizedCrop(224)],
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders(path, path=path, bs=64)

dls_test = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=RandomSplitter(valid_pct=0.5, seed=0),
    get_items=get_image_files, get_y=parent_label,
    item_tfms=[RandomResizedCrop(224)],
    batch_tfms=Normalize.from_stats(*imagenet_stats)
).dataloaders('RESISC45', path='RESISC45', bs=64)

timm_model = timm.create_model('MODEL_NAME', pretrained=True, num_classes=30)
for model in [timm_model]:
    print('model',model)
    learn = Learner(dls, model, metrics=[accuracy,top_k_accuracy]).to_fp16()
    with learn.distrib_ctx(): 
       learn.fit_flat_cos(300, 1e-3)
    with learn.distrib_ctx(): 
       learn.validate(dl=dls.valid)
    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.confusion_matrix())
    learn_test = Learner(dls_test, model,  metrics=accuracy).to_fp16()
    with learn_test.distrib_ctx(): 
       learn_test.validate(dl=dls_test.valid)
    interp = ClassificationInterpretation.from_learner(learn_test)
    print(interp.confusion_matrix())
```


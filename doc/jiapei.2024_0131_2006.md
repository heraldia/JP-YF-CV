sav ~/Downloads/jiapei.2024_0131_2006.md


http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
train_vp_detection.py 

[done] put it under the InMeMo/ path, rename to pascal-5i.


conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit -c pytorch -c conda-forge
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -c conda-forge  # 2024_0131_2033 Y
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -c conda-forge

# 2024_0131_2038 
rename VOCdevkit -> pascal-5i


feature_extractor_folderwise_segmentation.py < == > train_vp_segmentation.py

# 2024_0131_2059 
python tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val
cuda


# todo-jiapei on cuda machine
https://github.com/Jackieam/InMeMo/blob/master/Segmentation.md
    https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b_ft_in12k/tree/main -> program downloads
python tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val
to check if folder [features_vit-laion2b_pixel-level-val] contains new files.
-> './pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn/folder3_top_50-similarity.json'

    folder0_top_50-similarity.json
    folder1_top_50-similarity.json
    folder2_top_50-similarity.json
    folder3_top_50-similarity.json

    /Users/y0f00k5/Documents/githubProject/jiapei-InMeMo/evaluate/splits/pascal/trn/fold0.txt   split
    2007_000032__01
    2007_000256__01

    --fold 3  // fold == folder == split

    https://github.com/Jackieam/InMeMo/tree/master/evaluate/splits/pascal/trn



train_vp_segmentation.py <==> trainer/val_pascal_dataloader.py


python train_vp_segmentation.py --mode spimg_spmask --output_dir output_samples --fold 3 --device cuda:0 --base_dir ./pascal-5i --batch-size 32 --lr 40 --epoch 100 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model pad --p-eps 1
python train_vp_segmentation.py --mode spimg_spmask --output_dir output_samples --fold 3 --device cpu --base_dir ./pascal-5i --batch-size 32 --lr 40 --epoch 100 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model pad --p-eps 1
python train_vp_segmentation.py --mode spimg_spmask --output_dir output_samples --fold 3 --base_dir ./pascal-5i --batch-size 32 --lr 40 --epoch 100 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model pad --p-eps 1

python tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val


'./pascal-5i/VOC2012/features_vit-laion2b_pixel-level_trn/folder3_top_50-similarity.json'

---------


https://github.com/smhassanerfani/atlantis/tree/master/adk/dataset/s3/river
..
images
masks
rgb_masks
annotations.xml
flickr.json


# todo-jiapei on cuda machine
pascal-5i/VOC2012 -> river
pascal -> river

python tools/feature_extractor_folderwise_segmentation.py vit_large_patch14_clip_224.laion2b features_vit-laion2b_pixel-level val

river
    folder0_top_50-similarity.json

python train_vp_segmentation.py --mode spimg_spmask --output_dir output_samples --fold 0 --device cuda:0 --base_dir ./river --batch-size 32 --lr 40 --epoch 100 --scheduler cosinewarm --optimizer Adam --arr a1 --vp-model pad --p-eps 1

river/
    JPEGImages
    SegmentationClass
    SegmentationObject

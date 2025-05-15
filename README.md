# Serialized attention and masked residual network structure for free-form image inpainting

The paper can be found [here](https://doi.org/10.1016/j.knosys.2025.113385).

Please run the codes where the python is Version 3.x and pytorch>=0.4.

## Preparation
Before running the codes, you should prepare training/evaluation image file list (flist) and mask file list (flist). You can refer to the folowing command to generate .flist file:
```
python flist.py --path your_dataset_folder --output xxx.flist
```

## Training
There are some hyperparameters that you can adjust in the main.py. To train the model, you can run:
```
python main.py --bs 4 --gpus 2 --prefix XX --img_flist your_training_images.flist --mask_flist your_training_masks.flist
```
PS: You can set the "--bs" and "--gpus" to any number as you like. The above is just an example.

## Evaluation
To evaluate the model, you can use GPU or CPU to run.

For GPU:
```
python eval.py --bs your_batch_size --model your_checkpoint_path --img_flist your_eval_images.flist --mask_flist your_eval_masks.flist
```

For CPU:
```
python eval.py --cpu --bs your_batch_size --model your_checkpoint_path --img_flist your_eval_images.flist --mask_flist your_eval_masks.flist
```

PS: The pretrained model under folder './pretrained_model/' is trained from Places2 dataset with [Irregular Mask](https://nv-adlr.github.io/publication/partialconv-inpainting) dataset. **Please train RN from scratch if you test data not from Places2 or using regular mask.**

## Cite Us
```
@misc{Ethereal2025,
    title={Serialized attention and masked residual network structure for free-form image inpainting},
    author={Jinlong TANG, Kai HE, Mingjuan Tian , Ziqi YANG, Shuai Yuan},
    journal={Knowledge-Based Systems},
    year={2025},
    publisher={Elsevier}
}
```

## Appreciation
The codes refer to [EdgeConnect](https://github.com/knazeri/edge-connect). Thanks for the authors of it！

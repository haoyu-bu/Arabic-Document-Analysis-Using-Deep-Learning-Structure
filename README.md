# Arabic-Document-Analysis-Using-Deep-Learning-Structure

Source codes of our CS591 project: Arabic Document Analysis Using Deep Learning Structure.

**Wenda Qin,** **Hao Yu**

## Faster-R-CNN-based method

## CRAFT-based method
The implementation of CRAFT-based method are in the folder craft and segmentation
### Requirements
- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- requirments.txt
```
pip install -r craft/requirements.txt
```

### Generating text detection results

- Download the trained [model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) which is trained on SynthText, IC13, IC17 dataset.
 
- Run with pretrained model
``` (with python 3.7)
python craft/test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```

The result images will be saved to `./result` by default.

### Segmentation

- Train
``` (with python 3.7)
python segmentation/main.py --train --input=[folder path to input images] --output=[folder path to output files]
```

- Test
``` (with python 3.7)
python segmentation/main.py --input=[folder path to input images] --output=[folder path to output files]
```

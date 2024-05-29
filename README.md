# DM2F-Net-improve

By Yuyan Chen.

## Results

<table border="1" style="width: 100%; text-align: center;">
  <tr>
    <th> </th>
    <th>算法</th>
    <th>MSE</th>
    <th>PNSR</th>
    <th>SSIM</th>
    <th>CIEDE2000</th>
  </tr>
  <tr>
   <th rowspan="2">O-Haze</th>
    <td>baseline</td>
    <td>0.0038</td>
    <td>24.304</td>
    <td>0.7192</td>
    <td>4.7643</td>
  </tr>
  <tr>
    <td>improve</td>
    <td>0.0037</td>
    <td>24.436</td>
    <td>0.7242</td>
    <td>4.7547</td>
  </tr>
  <tr>
    <th rowspan="2">HazeRD</th>
    <td>baseline</td>
    <td>0.0679</td>
    <td>14.481</td>
    <td>0.8314</td>
    <td>16.161</td>
  </tr>
  <tr>
    <td>improve</td>
    <td>0.0695</td>
    <td>14.584</td>
    <td>0.8315</td>
    <td>16.036</td>
  </tr>
</table>


The dehazing results can be found at [Baidu Wangpan](https://pan.baidu.com/s/1ajlm7uyo4cCjADaek6EnSw).

## Installation & Preparation

Make sure you have `Python>=3.7` installed on your machine.

**Environment setup:**

1. Create conda environment

       conda create -n dm2f
       conda activate dm2f

2. Install dependencies (test with PyTorch 1.8.0):

   1. Install pytorch==1.8.0 torchvision==0.9.0 (via conda, recommend).

   2. Install other dependencies

          pip install -r requirements.txt

* Prepare the dataset

   * Download the RESIDE dataset from the [official webpage](https://sites.google.com/site/boyilics/website-builder/reside).

   * Download the O-Haze dataset from the [official webpage](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

   * Make a directory `./data` and create a symbolic link for uncompressed data, e.g., `./data/RESIDE`.

## Training

1. Set the path of datasets in `tools/config.py`
2. Run by ```python train.py```

Use pretrained ResNeXt (resnext101_32x8d) from torchvision.

Training a model on a single RTX 3080 Ti(12GB) GPU takes about 6 hours.

## Testing

1. Set the path of five benchmark datasets in `tools/config.py`.
2. Put the trained model in `./ckpt/`.
3. Run by `python test.py` (for O-Haze/HazeRD/RESIDE) or `python output.py` (for own pictures)

*Settings* of testing were set at the top of `test.py`, and you can conveniently
change them as you need.

## License

DM2F-Net is released under the [MIT license](LICENSE).

## Citation

If you find the paper or the code helpful to your research, please cite the project.

```
@inproceedings{deng2019deep,
  title={Deep multi-model fusion for single-image dehazing},
  author={Deng, Zijun and Zhu, Lei and Hu, Xiaowei and Fu, Chi-Wing and Xu, Xuemiao and Zhang, Qing and Qin, Jing and Heng, Pheng-Ann},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2453--2462},
  year={2019}
}
```

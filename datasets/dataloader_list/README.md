本文件用于进行数据处理：
在数据集构建的过程中可以自行选择DAVIS、YouTube-Vos或者全部；

**Note**

`adaptor_dataset_.py` and  `adaptor_datase.py`的差异仅仅体现在数据增强的函数库。

`YTB_DAVIS_Dataset()`：
`split`: train/val
`datasets`: ["Davis"]/["YoutubeVOS"]/["YoutubeVOS","Davis"]
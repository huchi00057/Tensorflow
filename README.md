# 目錄
* [Tensorflow-GPU 版本配對一覽](#tensorflow-gpu-版本配對一覽)
    * [若想查看CUDA版本](#若想查看cuda版本) 
* [Tensorflow-GPU 安裝](#tensorflow-gpu-安裝)
* [Tensorflow-GPU 測試指令](#tensorflow-gpu-測試指令)
* [查詢Tensorflow版本](#查詢tensorflow版本)
* [TensorLayer 介紹](#tensorlayer-介紹)
* [安裝 TensorLayer](#安裝-tensorlayer)

# Tensorflow-GPU 版本配對一覽
|    Tensorflow 版本    | Python 版本 | cuDNN | CUDA |
|----------------------|-------------|-------|------|
| tensorflow_gpu-2.6.0 | 3.6到3.9 | 8.1 | 11.2 |
| tensorflow_gpu-2.5.0 | 3.6到3.9 | 8.1 | 11.2 |
| tensorflow_gpu-2.4.0 | 3.6到3.8 | 8.0 | 11.0 |
| tensorflow_gpu-2.3.0 | 3.5到3.8 | 7.6 | 10.1 |
| tensorflow_gpu-2.2.0 | 3.5到3.8 | 7.6 | 10.1 |
| tensorflow_gpu-2.1.0 | 3.5到3.7 | 7.6 | 10.1 |
| tensorflow_gpu-2.0.0 | 3.5到3.7 | 7.4 | 10.0 |

更久遠版本配對詳見🔥🔥官網 [點我](https://tensorflow.google.cn/install/source_windows#gpu)

## 若想查看CUDA版本
    nvcc -V
![image](https://user-images.githubusercontent.com/46515944/198818059-605804d4-630e-4c35-b359-0ec494ee6207.png)

# Tensorflow-GPU 安裝
    pip install tensorflow-gpu    
 如果想指定版本可輸入
 
    pip install tensorflow-gpu==2.6.0
 

# Tensorflow-GPU 測試指令
    python
    import tensorflow as tf
    tf.test.is_gpu_available()
 
 如果回傳 True(以及顯卡資訊，像我就是 NVIDIA GeForce RTX 2070)，代表有安裝好，反之 False，就是沒裝好。可能前面再安裝就有出狀況，或是版本配對不吻合等問題。
 
![198817447-b94f94fd-d55d-4956-a570-f840df667866](https://user-images.githubusercontent.com/46515944/198817536-a89a44d7-3ebe-4c36-a681-2ff15e9ecff9.png)

# 查詢Tensorflow版本
    python 
    import tensorflow as tf
    tf.__version__
    
![image](https://user-images.githubusercontent.com/46515944/198817509-a70bb29f-d3e1-4abf-bc3f-4f5f1c714ef4.png)

# TensorLayer 介紹
📌TensorLayer 是基於 Tensorflow 的深度學習與強學習庫，特別設計給研究人員與工程師使用的。

📌目前版本支持TensorFlow、Pytorch、MindSpore、PaddlePaddle、OneFlow 和 Jittor 作為後端，允許用戶在 Nvidia-GPU 和華為-Ascend 等不同硬件上運行代碼。

📌優點：
1. Simplicity(簡單)：人家用好好的，你直接套用就好
2. Flexibility(彈性)：什麼資料都公開出來，看你要改啥都很彈性
3. Zero-cost Abstraction(零成本抽象)：除了輕易上手超級讚，你還不需要付錢

📌它已被世界各地的研究人員和工程師使用，包括來自北京大學、倫敦帝國理工學院、加州大學伯克利分校、卡內基梅隆大學、斯坦福大學以及谷歌、微軟、阿里巴巴、騰訊、小米和彭博等公司的研究人員和工程師。

📌詳細資料目前有英文跟中文
[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayer.readthedocs.io/)
[![Chinese Documentation](https://img.shields.io/badge/documentation-%E4%B8%AD%E6%96%87-blue.svg)](https://tensorlayercn.readthedocs.io/)
[![Chinese Book](https://img.shields.io/badge/book-%E4%B8%AD%E6%96%87-blue.svg)](http://www.broadview.com.cn/book/5059/)

以上資訊來自於🔥🔥 [Tensorflow官方Github](https://github.com/tensorlayer/tensorlayer)

# 安裝 TensorLayer
    pip isntall tensorlayer

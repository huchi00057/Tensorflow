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

更久遠版本配對詳見官網 [點我](https://tensorflow.google.cn/install/source_windows#gpu)

# Tensorflow-GPU 測試指令
    python
    import tensorflow as tf
    tf.test.is_gpu_available()
 
 如果回傳 True(以及顯卡資訊，像我就是 NVIDIA GeForce RTX 2070)，代表有安裝好，反之 False，就是沒裝好。可能前面再安裝就有出狀況，或是版本配對不吻合等問題。
 
![198817447-b94f94fd-d55d-4956-a570-f840df667866](https://user-images.githubusercontent.com/46515944/198817536-a89a44d7-3ebe-4c36-a681-2ff15e9ecff9.png)

# 查詢Tensorflow版本
    python import tensorlfow as tf
    tf.__version__
    
![image](https://user-images.githubusercontent.com/46515944/198817509-a70bb29f-d3e1-4abf-bc3f-4f5f1c714ef4.png)



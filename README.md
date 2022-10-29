# Tensorflow-GPU 版本配對一覽

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



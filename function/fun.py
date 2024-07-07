import numpy as np
import pandas as pd
import winsound as sd
import tensorflow as tf #텐서플로우

### 대회기준_모델정확도
def csi(data):
    num = data[0][0]
    all_sum = np.sum(data)

    # 대각선 요소들의 합 계산
    diagonal_elements = np.diagonal(data)
    diagonal_sum = np.sum(diagonal_elements)
    
    result = (diagonal_sum-num)/(all_sum+num)

    return result

def csi_metric(y_true, y_pred):
    # y_pred를 예측 클래스 인덱스로 변환
    y_pred_classes = tf.argmax(y_pred, axis=-1)

    # confusion matrix 계산
    cm = tf.math.confusion_matrix(y_true, y_pred_classes, num_classes=10)

    # confusion matrix에서 필요한 값 추출
    num = cm[0, 0]
    all_sum = tf.reduce_sum(cm)

    # 대각선 요소들의 합 계산
    diagonal_elements = tf.linalg.diag_part(cm)
    diagonal_sum = tf.reduce_sum(diagonal_elements)
    
    result = (diagonal_sum - num) / (all_sum + num)

    return result

### train_test_split
def idx_find(df):
    idx = df[df['dh'] == 3].index
    idx = idx-1
    idx = idx[1:]
    temp = df.iloc[idx]
    aa = temp.iloc[:int(len(idx) *0.6)].index[-1]
    bb = temp.iloc[:int(len(idx) *0.6) + int((len(idx)-int(len(idx) *0.5)) *0.5)].index[-1]

    return aa, bb

### 알림음
def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

### lr조정
def scheduler(epoch, lr):
    return float(lr * tf.math.exp(-0.1 ** epoch))
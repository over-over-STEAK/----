import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

# -------------------------------------------------------------------------
# 算子融合實作之一：高效率 Fused RMSNorm (針對向量層級)
# -------------------------------------------------------------------------
def fused_rmsnorm(x_list, gamma_list, eps=1e-5):
    """
    x_list: List[Value], 輸入向量
    gamma_list: List[Value], 縮放權重向量
    """
    d = len(x_list)
    x_raw = [v.data for v in x_list]
    gamma_raw = [v.data for v in gamma_list]
    
    # 1. 前向傳播 (Forward Pass) - 使用純數值計算，不建立中間 Value 節點
    ss = sum(x**2 for x in x_raw)
    rms = math.sqrt(ss / d + eps)
    
    x_hat = [x / rms for x in x_raw]
    y_raw = [xh * g for xh, g in zip(x_hat, gamma_raw)]
    
    # 建立輸出節點包裝，將所有輸入與參數設為 _children
    out_list = [Value(y, _children=(x_list[i], gamma_list[i]), _op='RMSNorm') for i, y in enumerate(y_raw)]
    
    # 2. 反向傳播 (Backward Pass) - 手寫手算梯度
    def _backward():
        # 收集上游傳回來的梯度
        dy = [out.grad for out in out_list]
        
        # 計算共通的項： sum( dy[j] * gamma[j] * x[j] )
        sum_dy_gamma_x = sum(dy[j] * gamma_raw[j] * x_raw[j] for j in range(d))
        
        # 更新 gamma 的梯度
        for i in range(d):
            gamma_list[i].grad += dy[i] * x_hat[i]
            
        # 更新 x 的梯度 (套用推導出的高效向量化公式)
        term2_scale = 1.0 / (d * (rms ** 3))
        for i in range(d):
            dx_i = (gamma_raw[i] / rms) * dy[i] - (x_raw[i] * term2_scale * sum_dy_gamma_x)
            x_list[i].grad += dx_i

    # 將 backward 函式綁定到輸出節點上
    # 為了確保拓撲排序能正確觸發，我們讓這組輸出共享同一個 backward 行為
    for out in out_list:
        out._backward = _backward
        
    return out_list

# -------------------------------------------------------------------------
# 算子融合實作之二：高效率 Fused 數值穩定 Softmax
# -------------------------------------------------------------------------
def fused_softmax(x_list):
    """
    x_list: List[Value], 輸入的 Logits 向量
    """
    x_raw = [v.data for v in x_list]
    
    # 1. 前向傳播 - 數值穩定防溢位
    max_val = max(x_raw)
    exps = [math.exp(x - max_val) for x in x_raw]
    sum_exps = sum(exps)
    s_raw = [exp / sum_exps for exp in exps]
    
    # 建立輸出節點
    out_list = [Value(s, _children=(x_list[i],), _op='Softmax') for i, s in enumerate(s_raw)]
    
    # 2. 反向傳播
    def _backward():
        ds = [out.grad for out in out_list]
        
        # 計算內積 sum( ds[j] * s[j] )
        sum_ds_s = sum(ds[j] * s_raw[j] for j in range(len(x_list)))
        
        # 套用 Softmax 梯度公式
        for i in range(len(x_list)):
            dx_i = s_raw[i] * (ds[i] - sum_ds_s)
            x_list[i].grad += dx_i

    for out in out_list:
        out._backward = _backward
        
    return out_list
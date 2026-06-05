import torch
from measure import app

if __name__ == "__main__":
        # 格子数
    n = 62
    # レイノルズ数
    Re = 100
    # 蓋の速度
    u0 = 1.0
    # 刻み時間
    dt = 0.001
    # スクリーンショット間隔
    shot = 50
    # 計算精度
    dtype = torch.float64

    app(n=n, Re=Re, u0=u0, dt=dt, shot=shot, dtype=dtype)

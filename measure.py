import os
import torch
import matplotlib.pyplot as plt
from cpu import CpuClass
from gpu import GpuClass
from gpu_triton import TritonClass
import numpy as np
import time

# アニメーション用の PNG ファイルを作成する
def draw_fig(ax, name, val, n, t):
    ax.cla()

    # 内部圧力 (n×n): [x,y]インデックス
    p_int = val[2][1:-1, 1:-1]
    pmax_abs = float(np.abs(p_int).max())
    if pmax_abs < 1e-10:
        pmax_abs = 1.0

    # MAC 面速度をセル中心に補間 (n×n), [x,y]インデックス
    u_c = (val[0][1:-2, 1:-1] + val[0][2:-1, 1:-1]) / 2
    v_c = (val[1][1:-1, 1:-2] + val[1][1:-1, 2:-1]) / 2

    u_max = float(np.abs(u_c).max())
    v_max = float(np.abs(v_c).max())
    print(f"[{name} t={t}] p_max={pmax_abs:.3e}, u_max={u_max:.3e}, v_max={v_max:.3e}")

    # pcolormesh: [x,y] → 転置して [y,x]、セル境界を整数座標で指定
    xi = np.arange(n + 1, dtype=float)
    yi = np.arange(n + 1, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    ax.pcolormesh(XI, YI, p_int.T, cmap='RdBu_r', vmin=-pmax_abs, vmax=pmax_abs)

    # 矢印の間引き
    step = max(1, n // 12)
    sample = np.arange(0, n, step)
    xc = sample + 0.5
    yc = sample + 0.5
    X, Y = np.meshgrid(xc, yc)          # [y,x]配置
    u_q = u_c[np.ix_(sample, sample)].T # [y,x]
    v_q = v_c[np.ix_(sample, sample)].T # [y,x]
    spd_max = float(np.hypot(u_q, v_q).max())
    if spd_max > 1e-10:
        ax.quiver(X, Y, u_q, v_q, color='k',
                  scale=spd_max / (step * 2), scale_units='xy', angles='xy',
                  linewidth=1.5)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.set_title(f"fluid velocities: {name}, t={t}")
    plt.savefig(f"images/{name}_{t:06d}.png", bbox_inches='tight')

# メイン関数
def app(n, Re, u0, shot, dtype):
    # 各実行クラスのインスタンスを作成する
    d = dict()
    d["cpu"] = CpuClass(n=n, Re=Re, u0=u0, dt=dt, dtype=dtype)
    d["gpu"] = GpuClass(n=n, Re=Re, u0=u0, dt=dt, dtype=dtype)
    d["triton"] = TritonClass(n=n, Re=Re, u0=u0, dt=dt, dtype=dtype)

    # 時間発展させる
    os.makedirs("images", exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 10))

    # 最初は warm-up
    print("Warming up...")
    for name, inst in d.items():
        for t in range(max(1, int(1e-5 / dt))):
            inst.one_loop()

    
    print(f"N={n}, Re={Re}, dt={dt}, dtype={dtype}")
    for name, inst in d.items():
        inst.clear()
        shot_count = 0
        inst.synchronize()
        start_time = time.time()
        for t in range(max(1, int(1e-5 / dt))):
            inst.one_loop()
            if t % shot == 0:
                val = inst.to_cpu()
                shot_count += 1
                draw_fig(ax=ax, name=name, val=val, n=n, t=shot_count)
        inst.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"calculated {name}: elapsed:{elapsed} sec")
    
# メイン関数呼び出し
if __name__ == "__main__":
    # 格子数
    ns = [50, 100, 150]

    # レイノルズ数
    Res = [50, 50, 50]

    # 蓋の速度
    u0s = [1.0, 1.0, 1.0]

    # 刻み時間
    dts = [1e-6, 1e-6, 1e-6]

    # スクリーンショット間隔
    shots = [1, 1, 1]

    # 計算精度
    dtypes = [torch.float64, torch.float64, torch.float64]
    #dtypes = [torch.float32, torch.float32, torch.float32]

    for n, Re, u0, dt, shot, dtype in zip(ns, Res, u0s, dts, shots, dtypes):
        app(n, Re, u0, shot, dtype)

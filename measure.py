import matplotlib.pyplot as plt
from cpu import CpuClass
from gpu import GpuClass
from gpu_triton import TrironClass
import numpy as np
import time

# アニメーション用の PNG ファイルを作成する
def draw_fig(ax, name, val, n, t):
    x, y = np.meshgrid(np.arange(0, n + 1), np.arange(0, n + 1))
    u = val[0][1:n+2,1:n+2] * 15.0
    v = val[1][1:n+2,1:n+2] * 15.0
    pmin = np.min(np.abs(val[2]))
    pmax = np.max(np.abs(val[2]) - pmin)
    pcol = (np.abs(val[2]) - pmin) / pmax

    plt.rc('font', size=22)
    plt.xlim(0, n)
    plt.ylim(0, n)
    ax.imshow(pcol, origin='lower')
    ax.quiver(x, y, u, v, units='xy', scale=1, color='black')
    ax.grid()
    plt.title("fluid velocities:" + name + ", t=" + str(t))
    figname = "images/{}_{:06d}.png".format(name, t)
    plt.savefig(figname)

# メイン関数
def app(n, Re, u0, dt, shot, dtype):
    # 各実行クラスのインスタンスを作成する
    d = dict()
    d["cpu"] = CpuClass(n=n, Re=Re, u0=u0, dt=dt, dtype=dtype)
    d["gpu"] = GpuClass(n=n, Re=Re, u0=u0, dt=dt, dtype=dtype)
    d["triton"] = TrironClass(n=n, Re=Re, u0=u0, dt=dt, dtype=dtype)

    # 時間発展させる
    fig, ax = plt.subplots(figsize=(20, 10))
    for name, inst in d.items():
        shot_count = 0
        start_time = time.time()
        for t in range(int(100/dt)):
            inst.one_loop()
            if t % shot == 0:
                val = inst.to_cpu()
                shot_count += 1
                draw_fig(ax=ax, name=name, val=val, n=n, t=shot_count)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"calculated {name}: elapsed:{elapsed} msec")
    
# メイン関数呼び出し
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
    dtype = np.float64

    app(n=n, Re=Re, u0=u0, dt=dt, shot=shot, dtype=dtype)

import torch

class BaseClass:
    def __init__(self, n, Re, u0, dt, dtype , device):
        # CUDA が利用できることを確認する
        if device == "cuda":
            assert torch.cuda.is_available()

        # 初期値
        self.n = n
        self.d = 1/n
        self.re = Re
        self.u0 = u0
        self.dt = dt
        self.dtype = dtype
        self.device = device
        self.epsilon = 1e-6

        # 格子作成
        # p は格子内側の中心に定義されるため、行や列の数が2少ないことに注意
        self.p = torch.zeros(n + 2, n + 2, dtype=dtype, device=device)
        self.u = torch.zeros(n + 2, n + 2, dtype=dtype, device=device)
        self.v = torch.zeros(n + 2, n + 2, dtype=dtype, device=device)

        # NS 方程式の右辺計算式のメモリ確保
        self.p_rhs = torch.zeros(n, n, dtype=dtype, device=device)
        self.u_rhs = torch.zeros(n, n, dtype=dtype, device=device)
        self.v_rhs = torch.zeros(n, n, dtype=dtype, device=device)

        # 補助変数のメモリ確保
        self.ld = torch.zeros(n, n, dtype=dtype, device=device)
        self.ub = torch.zeros(n, n, dtype=dtype, device=device)
        self.vb = torch.zeros(n, n, dtype=dtype, device=device)
        self.p_uv = torch.zeros(n, n, dtype=dtype, device=device)

    # 各辺の境界条件設定
    def set_boundary_vert(self, src, dst):
        self.u[dst, 1:-1] =  self.u[src, 1:-1]
        self.v[dst, 1:-1] = -self.v[src, 1:-1]
        if src > dst:
            self.p[dst, 1:-1] = self.p[src - 1, 1:-1] + (self.u[src + 2, 1:-1] - 2 * self.u[src + 1, 1:-1]) / self.d / self.re
        else:
            self.p[dst - 1, 1:-1] = self.p[src + 1, 1:-1] + (self.u[src - 2, 1:-1] - 2 * self.u[src - 1, 1:-1]) / self.d / self.re

    def set_boundary_horiz(self, src, dst, u0):
        self.u[1:-1, dst] = u0 - self.u[1:-1, src]
        self.v[1:-1, dst] =  self.v[1:-1, src]
        if src > dst:
            self.p[1:-1, dst] = self.p[1:-1, src - 1] + (self.v[1:-1, src + 2] - 2 * self.v[1:-1, src + 1]) / self.d / self.re
        else:
            self.p[1:-1, dst - 1] = self.p[1:-1, src + 1] + (self.v[1:-1, src - 2] - 2 * self.v[1:-1, src - 1]) / self.d / self.re

    # 境界条件の設定
    def set_boundary(self):
        # 辺DA
        self.set_boundary_vert(2, 0)
        # 辺AB
        self.set_boundary_horiz(2, 0, 0)
        # 辺BC
        self.set_boundary_vert(self.n - 1, self.n + 1)
        # 辺DC
        self.set_boundary_horiz(self.n - 1, self.n + 1, self.u0)

    # 補助変数の計算
    # ub/vb/ld/p_uv は行や列が2少ないことに注意 
    def calc_aux(self):
        self.ub[:, :] = (self.u[2:, 2:] + self.u[2:,1:-1] + self.u[1:-1, 2:] + self.u[1:-1, 1:-1]) / 4
        self.vb[:, :] = (self.v[2:, 2:] + self.v[2:,1:-1] + self.v[1:-1, 2:] + self.v[1:-1, 1:-1]) / 4
        self.ld[:, :] = (self.u[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.d + (self.v[1:-1, 2:] - self.v[1:-1, 1:-1]) / self.d
        self.p_uv[:, :] = (
            (self.u[2:, 1:-1] - self.u[1:-1, 1:-1]) * (self.u[2:, 1:-1] - self.u[1:-1, 1:-1])
            + (self.v[1:-1, 2:] - self.v[1:-1, 1:-1]) * (self.v[1:-1, 2:] - self.v[1:-1, 1:-1])
            + (self.u[1:-1, 2:] - self.u[1:-1, :-2]) * (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / 2
            - self.ld[:, :] * self.d * self.d / self.dt
        )

    # pressure を計算
    def calc_pressure(self):
        self.p_rhs[:, :] = 0.25 * (self.p[2:, 1:-1] + self.p[:-2,1:-1] + self.p[1:-1,2:] + self.p[1:-1,:-2] + self.p_uv[:,:])
        er = torch.max(torch.abs(self.p[1:-1, 1:-1] - self.p_rhs[:, :])).item()
        return er

    # Poisson 方程式を Gauss の陰解法で解く
    def solve_poisson(self):
        counter = 0
        er = float('inf')
        while er > self.epsilon:
            er = self.calc_pressure()
            self.p[1:-1, 1:-1] = self.p_rhs[:, :]
            counter += 1
            if counter > 3000:
                raise Exception("Gauss 法が収束しませんでした")

    # u の時間発展
    def evolve_u(self):
        self.u_rhs[:, :] = self.u[1:-1, 1:-1] * (self.u[2:,1:-1] - self.u[:-2, 1:-1]) / 2 / self.d + self.vb[:, :] * (self.u[1:-1,2:] - self.u[1:-1, :-2]) / 2 / self.d + (self.p[2:,1:-1] - self.p[1:-1, 1:-1]) / self.d - (1 / self.re / self.d / self.d) * (self.u[2:,1:-1] + self.u[:-2,1:-1] + self.u[1:-1, 2:] + self.u[1:-1, :-2] - 4 * self.u[1:-1, 1:-1])

    # v の時間発展
    def evolve_v(self):
        self.v_rhs[:, :] = self.v[1:-1, 1:-1] * (self.v[1:-1,2:] - self.v[1:-1, :-2]) / 2 / self.d + self.ub[:, :] * (self.v[2:,1:-1] - self.v[:-2, 1:-1]) / 2 / self.d + (self.p[1:-1,2:] - self.p[1:-1, 1:-1]) / self.d - (1 / self.re / self.d / self.d) * (self.v[2:,1:-1] + self.v[:-2,1:-1] + self.v[1:-1, 2:] + self.v[1:-1, :-2] - 4 * self.v[1:-1, 1:-1])

    # 時間発展を計算
    def evolve(self):
        self.u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - self.dt * self.u_rhs[:, :]
        self.v[1:-1, 1:-1] = self.v[1:-1, 1:-1] - self.dt * self.v_rhs[:, :]

    # 時間刻みの計算ループを構成
    def one_loop(self):
        self.set_boundary()
        self.calc_aux()
        self.solve_poisson()
        self.evolve_u()
        self.evolve_v()
        self.evolve()

    # 物理量を CPU に転送
    def to_cpu(self):
        return (self.u, self.v, self.p)
import math

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
        self.epsilon = 1e-3

        # 格子作成
        # p は格子内側の中心に定義されるため、行や列の数が2少ないことに注意
        self.p = torch.zeros(n + 2, n + 2, dtype=dtype, device=device)
        self.u = torch.zeros(n + 3, n + 2, dtype=dtype, device=device)
        self.v = torch.zeros(n + 2, n + 3, dtype=dtype, device=device)

        # NS 方程式の右辺計算式のメモリ確保
        self.p_rhs = torch.zeros(n, n, dtype=dtype, device=device)
        self.u_rhs = torch.zeros(n, n, dtype=dtype, device=device)
        self.v_rhs = torch.zeros(n, n, dtype=dtype, device=device)

        # 補助変数のメモリ確保
        self.ld = torch.zeros(n, n, dtype=dtype, device=device)
        self.ub = torch.zeros(n, n, dtype=dtype, device=device)
        self.vb = torch.zeros(n, n, dtype=dtype, device=device)
        self.p_uv = torch.zeros(n, n, dtype=dtype, device=device)

    def set_boundary_da(self):
        self.p[0, :] = self.p[1, :] + (self.u[3, :] - 2.0 * self.u[2, :]) / self.d / self.re
        self.u[0, :] = self.u[2, :]
        self.u[1, :] = 0
        self.v[0, :] = -self.v[1, :]

    def set_boundary_ab(self):
        self.p[:, 0] = self.p[:, 1] + (self.v[:, 3] - 2.0 * self.v[:, 2]) / self.d / self.re
        self.v[:, 0] = self.v[:, 2]
        self.v[:, 1] = 0
        self.u[:, 0] = -self.u[:, 1]

    def set_boundary_bc(self):
        self.p[self.n + 1, :] = self.p[self.n, :] - (self.u[self.n - 1, :] - 2.0 * self.u[self.n, :]) / self.d / self.re
        self.u[self.n + 2, :] = self.u[self.n, :]
        self.u[self.n + 1, :] = 0
        self.v[self.n + 1, :] = -self.v[self.n, :]

    def set_boundary_dc(self):
        self.p[:, self.n + 1] = self.p[:, self.n] - (self.v[:, self.n - 1] - 2.0 * self.v[:, self.n]) / self.d / self.re
        self.v[:, self.n + 2] = self.v[:, self.n]
        self.v[:, self.n + 1] = 0
        self.u[:, self.n + 1] = 2.0 * self.u0 - self.u[:, self.n]

    # 境界条件の設定
    def set_boundary(self):
        # 辺DA
        self.set_boundary_da()
        # 辺AB
        self.set_boundary_ab()
        # 辺BC
        self.set_boundary_bc()
        # 辺DC
        self.set_boundary_dc()
    # 補助変数の計算
    # ub/vb/ld/p_uv は行や列が2少ないことに注意 
    def calc_aux(self):
        self.ub[:, :] = (self.u[2:-1, 2:] + self.u[2:-1,1:-1] + self.u[1:-2, 2:] + self.u[1:-2, 1:-1]) / 4
        self.vb[:, :] = (self.v[2:, 2:-1] + self.v[2:,1:-2] + self.v[1:-1, 2:-1] + self.v[1:-1, 1:-2]) / 4
        du = self.u[2:-1, 1:-1] - self.u[1:-2, 1:-1]
        dv = self.v[1:-1, 2:-1] - self.v[1:-1, 1:-2]
        self.ld[:, :] = du / self.d + dv / self.d
        self.p_uv[:, :] = (
            du.square()
            + dv.square()
            + du * dv / 2.0
            - self.ld * self.d * self.d / self.dt
        )

    # pressure を計算
    def calc_pressure(self):
        self.p_rhs[:, :] = 0.25 * (self.p[2:, 1:-1] + self.p[:-2,1:-1] + self.p[1:-1,2:] + self.p[1:-1,:-2] + self.p_uv[:,:])
        er = torch.max(torch.abs(self.p[1:-1, 1:-1] - self.p_rhs[:, :])).item()
        return er

    # Poisson 方程式を SOR 法で解く
    def solve_poisson(self):
        omega = 2.0 / (1.0 + math.sin(math.pi / self.n))
        i = torch.arange(self.n, device=self.device)[:, None]
        j = torch.arange(self.n, device=self.device)[None, :]
        red_mask = (i + j) % 2 == 0
        black_mask = ~red_mask
        counter = 0
        er = float('inf')
        while er > self.epsilon:
            p_old = self.p[1:-1, 1:-1].clone()

            self.set_boundary()
            self.calc_pressure()
            p_inner = self.p[1:-1, 1:-1]
            p_inner[red_mask] = (1.0 - omega) * p_inner[red_mask] + omega * self.p_rhs[red_mask]

            self.set_boundary()
            self.calc_pressure()
            p_inner = self.p[1:-1, 1:-1]
            p_inner[black_mask] = (1.0 - omega) * p_inner[black_mask] + omega * self.p_rhs[black_mask]

            p_inner -= p_inner.mean()
            er = torch.max(torch.abs(self.p[1:-1, 1:-1] - p_old)).item()
            counter += 1
            if counter > 5000:
                raise RuntimeError("SOR solver did not converge")
        self.set_boundary()

    # u の時間発展
    def evolve_u(self):
        self.u_rhs[:, :] = self.u[1:-2, 1:-1] * (self.u[2:-1,1:-1] - self.u[:-3, 1:-1]) / 2 / self.d + self.vb[:, :] * (self.u[1:-2,2:] - self.u[1:-2, :-2]) / 2 / self.d + (self.p[2:,1:-1] - self.p[1:-1, 1:-1]) / self.d - (1 / self.re / self.d / self.d) * (self.u[2:-1,1:-1] + self.u[:-3,1:-1] + self.u[1:-2, 2:] + self.u[1:-2, :-2] - 4 * self.u[1:-2, 1:-1])

    # v の時間発展
    def evolve_v(self):
        self.v_rhs[:, :] = self.v[1:-1, 1:-2] * (self.v[1:-1,2:-1] - self.v[1:-1, :-3]) / 2 / self.d + self.ub[:, :] * (self.v[2:,1:-2] - self.v[:-2, 1:-2]) / 2 / self.d + (self.p[1:-1,2:] - self.p[1:-1, 1:-1]) / self.d - (1 / self.re / self.d / self.d) * (self.v[2:,1:-2] + self.v[:-2,1:-2] + self.v[1:-1, 2:-1] + self.v[1:-1, 1:-2] - 4 * self.v[1:-1, 1:-2])

    # 時間発展を計算
    def evolve(self):
        self.u[1:-2, 1:-1] = self.u[1:-2, 1:-1] - self.dt * self.u_rhs[:, :]
        self.v[1:-1, 1:-2] = self.v[1:-1, 1:-2] - self.dt * self.v_rhs[:, :]

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
        return (self.u.numpy(), self.v.numpy(), self.p.numpy())

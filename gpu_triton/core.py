import torch
import triton
import triton.language as tl
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gpu import GpuClass

@triton.jit
def calc_aux_ub(
    ub_ptr, u_ptr,
    stride_ub_m, stride_ub_n,
    stride_u_m, stride_u_n,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = row_offset[:, None]
    c = col_offset[None, :]

    mask = (r < M - 2) & (c < N - 2)
    u_r = r + 1
    u_c = c + 1

    p11 = tl.load(u_ptr + (u_r + 1) * stride_u_m + (u_c + 1) * stride_u_n, mask = mask, other = 0.0)
    p10 = tl.load(u_ptr + (u_r + 1) * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    p01 = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c + 1) * stride_u_n, mask = mask, other = 0.0)
    p00 = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)

    # 4点の平均を計算
    val = (p11 + p10 + p01 + p00) / 4.0

    # 出力域へ保存
    tl.store(ub_ptr + stride_ub_m * r + stride_ub_n * c, val, mask = mask)

@triton.jit
def calc_aux_ld(
    u_ptr, v_ptr, ld_ptr,
    stride_u_m, stride_u_n,
    stride_v_m, stride_v_n,
    stride_ld_m, stride_ld_n,
    d,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = row_offset[:, None]
    c = col_offset[None, :]

    mask = (r < M - 2) & (c < N - 2)
    u_r = r + 1
    u_c = c + 1

    u10 = tl.load(u_ptr + (u_r + 1) * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    u00 = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    v01 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c + 1) * stride_v_n, mask = mask, other = 0.0)
    v00 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c)     * stride_v_n, mask = mask, other = 0.0)

    val = (u10 - u00) / d + (v01 - v00) / d

    tl.store(ld_ptr + r * stride_ld_m + c * stride_ld_n, val, mask = mask)

@triton.jit
def calc_aux_uv(
    u_ptr, v_ptr, ub_ptr, vb_ptr, ld_ptr, uv_ptr,
    stride_u_m, stride_u_n,
    stride_v_m, stride_v_n,
    stride_ub_m, stride_ub_n,
    stride_vb_m, stride_vb_n,
    stride_ld_m, stride_ld_n,
    stride_uv_m, stride_uv_n,
    d, dt,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = row_offset[:, None]
    c = col_offset[None, :]

    mask = (r < M - 2) & (c < N - 2)
    u_r = r + 1
    u_c = c + 1

    u10 = tl.load(u_ptr + (u_r + 1) * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    u00 = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    v01 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c + 1) * stride_v_n, mask = mask, other = 0.0)
    v00 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c)     * stride_v_n, mask = mask, other = 0.0)

    ub00 = tl.load(ub_ptr + r    * stride_ub_m + c    * stride_ub_n, mask = mask, other = 0.0)
    ub10 = tl.load(ub_ptr + u_r  * stride_ub_m + c    * stride_ub_n, mask = mask, other = 0.0)
    vb00 = tl.load(vb_ptr + r    * stride_vb_m + c    * stride_vb_n, mask = mask, other = 0.0)
    vb01 = tl.load(vb_ptr + r    * stride_vb_m + u_c  * stride_vb_n, mask = mask, other = 0.0)

    ld00 = tl.load(ld_ptr + r * stride_ld_m + c * stride_ld_n, mask = mask, other = 0.0)

    val = ((u10 - u00) * (u10 - u00)
           + (v01 - v00) * (v01 - v00)
           + (ub10 - ub00) * (vb01 - vb00)
           - ld00 * d * d / dt)

    tl.store(uv_ptr + r * stride_uv_m + c * stride_uv_n, val, mask = mask)

@triton.jit
def calc_p_right(
    p_ptr, u_ptr, v_ptr, ub_ptr, vb_ptr, ld_ptr, p_out_ptr,
    stride_p_m, stride_p_n,
    stride_u_m, stride_u_n,
    stride_v_m, stride_v_n,
    stride_ub_m, stride_ub_n,
    stride_vb_m, stride_vb_n,
    stride_ld_m, stride_ld_n,
    d, dt,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = row_offset[:, None]
    c = col_offset[None, :]

    mask = (r < M - 2) & (c < N - 2)
    u_r = r + 1
    u_c = c + 1

    p_p0 = tl.load(p_ptr + (u_r + 1) * stride_p_m + (u_c)     * stride_p_n, mask = mask, other = 0.0)
    p_0p = tl.load(p_ptr + (u_r)     * stride_p_m + (u_c + 1) * stride_p_n, mask = mask, other = 0.0)
    p_m0 = tl.load(p_ptr + (u_r - 1) * stride_p_m + (u_c)     * stride_p_n, mask = mask, other = 0.0)
    p_0m = tl.load(p_ptr + (u_r)     * stride_p_m + (u_c - 1) * stride_p_n, mask = mask, other = 0.0)

    ld_00 = tl.load(ld_ptr + r * stride_ld_m + c * stride_ld_n, mask = mask, other = 0.0)
    ld2 = d * d / dt * ld_00

    u10 = tl.load(u_ptr + (u_r + 1) * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    u00 = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c)     * stride_u_n, mask = mask, other = 0.0)
    v01 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c + 1) * stride_v_n, mask = mask, other = 0.0)
    v00 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c)     * stride_v_n, mask = mask, other = 0.0)

    ub_00 = tl.load(ub_ptr + r       * stride_ub_m + c       * stride_ub_n, mask = mask, other = 0.0)
    ub_0m = tl.load(ub_ptr + r       * stride_ub_m + (c - 1) * stride_ub_n, mask = mask, other = 0.0)
    vb_00 = tl.load(vb_ptr + r       * stride_vb_m + c       * stride_vb_n, mask = mask, other = 0.0)
    vb_m0 = tl.load(vb_ptr + (r - 1) * stride_vb_m + c       * stride_vb_n, mask = mask, other = 0.0)

    val = 0.25 * ((p_p0 + p_0p + p_m0 + p_0m)
                  - ld2
                  + (u10 - u00) * (u10 - u00)
                  + (v01 - v00) * (v01 - v00)
                  + 2.0 * (ub_00 - ub_0m) * (vb_00 - vb_m0))
    tl.store(p_out_ptr + u_r * stride_p_m + u_c * stride_p_n, val, mask = mask)

@triton.jit
def calc_max_abs_triton(
    p_ptr, max_ptr, n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offset < n
    x = tl.load(p_ptr + offset, mask = mask, other = 0.0)
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis = 0)
    tl.store(max_ptr + pid, max_val)

@triton.jit
def evolve_u(
    u_rhs_ptr, u_ptr, v_ptr, p_ptr, vb_ptr,
    stride_u_rhs_m, stride_u_rhs_n,
    stride_u_m, stride_u_n, stride_v_m, stride_v_n,
    stride_p_m, stride_p_n, stride_vb_m, stride_vb_n,
    d, dt, re, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r = row_offset[:, None]
    c = col_offset[None, :]

    mask = (r < M - 2) & (c < N - 2)
    u_r = r + 1
    u_c = c + 1

    u00 = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c)     * stride_u_n)
    up0 = tl.load(u_ptr + (u_r + 1) * stride_u_m + (u_c)     * stride_u_n)
    um0 = tl.load(u_ptr + (u_r - 1) * stride_u_m + (u_c)     * stride_u_n)
    u0p = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c + 1) * stride_u_n)
    u0m = tl.load(u_ptr + (u_r)     * stride_u_m + (u_c - 1) * stride_u_n)
    vb00 = tl.load(vb_ptr + r * stride_vb_m + c * stride_vb_n)
    p00 = tl.load(p_ptr + (u_r)     * stride_p_m + (u_c)     * stride_p_n)
    pp0 = tl.load(p_ptr + (u_r + 1) * stride_p_m + (u_c)     * stride_p_n)

    val = (u00 * (up0 - um0) / 2.0 / d
           + vb00 * (u0p - u0m) / 2.0 / d
           + (pp0 - p00) / d
           - 1.0 / re / d / d * (up0 + u0p + um0 + u0m - 4.0 * u00))
    tl.store(u_rhs_ptr + r * stride_u_rhs_m + c * stride_u_rhs_n, val, mask = mask)

@triton.jit
def evolve_v(
    v_rhs_ptr, u_ptr, v_ptr, p_ptr, ub_ptr,
    stride_v_rhs_m, stride_v_rhs_n,
    stride_u_m, stride_u_n, stride_v_m, stride_v_n,
    stride_p_m, stride_p_n, stride_ub_m, stride_ub_n,
    d, dt, re, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    r = row_offset[:, None]
    c = col_offset[None, :]

    mask = (r < M - 2) & (c < N - 2)
    u_r = r + 1
    u_c = c + 1

    v00 = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c)     * stride_v_n)
    vp0 = tl.load(v_ptr + (u_r + 1) * stride_v_m + (u_c)     * stride_v_n)
    vm0 = tl.load(v_ptr + (u_r - 1) * stride_v_m + (u_c)     * stride_v_n)
    v0p = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c + 1) * stride_v_n)
    v0m = tl.load(v_ptr + (u_r)     * stride_v_m + (u_c - 1) * stride_v_n)
    ub00 = tl.load(ub_ptr + r * stride_ub_m + c * stride_ub_n)
    p00 = tl.load(p_ptr + (u_r)     * stride_p_m + (u_c)     * stride_p_n)
    p0p = tl.load(p_ptr + (u_r)     * stride_p_m + (u_c + 1) * stride_p_n)

    val = (v00 * (vp0 - vm0) / 2.0 / d
           + ub00 * (v0p - v0m) / 2.0 / d
           + (p0p - p00) / d
           - 1.0 / re / d / d * (vp0 + v0p + vm0 + v0m - 4.0 * v00))
    tl.store(v_rhs_ptr + r * stride_v_rhs_m + c * stride_v_rhs_n, val, mask = mask)

class TrironClass(GpuClass):
    def __init__(self, n, Re, u0, dt, dtype):
        super().__init__(n, Re, u0, dt, dtype)

    def calc_aux(self):
        M, N = self.u.shape
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        calc_aux_ub[grid](self.ub, self.u,
            self.ub.stride(0), self.ub.stride(1), self.u.stride(0), self.u.stride(1),
            M, N, BLOCK_M=32, BLOCK_N=32)
        calc_aux_ub[grid](self.vb, self.v,
            self.vb.stride(0), self.vb.stride(1), self.v.stride(0), self.v.stride(1),
            M, N, BLOCK_M=32, BLOCK_N=32)
        calc_aux_ld[grid](self.u, self.v, self.ld,
            self.u.stride(0), self.u.stride(1), self.v.stride(0), self.v.stride(1),
            self.ld.stride(0), self.ld.stride(1), self.d, M, N, BLOCK_M=32, BLOCK_N=32)
        calc_aux_uv[grid](self.u, self.v, self.ub, self.vb, self.ld, self.p_uv,
            self.u.stride(0), self.u.stride(1), self.v.stride(0), self.v.stride(1),
            self.ub.stride(0), self.ub.stride(1), self.vb.stride(0), self.vb.stride(1),
            self.ld.stride(0), self.ld.stride(1),
            self.p_uv.stride(0), self.p_uv.stride(1),
            self.d, self.dt, M, N, BLOCK_M=32, BLOCK_N=32)

    def max_abs_xx(self, p, work, BLOCK_SIZE):
        calc_max_abs_triton[(work.shape[0],)](p, work, p.numel(), BLOCK_SIZE)
        return work.max()

    def solve_poisson(self):
        M, N = self.p.shape
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        p_new = torch.empty_like(self.p)
        delta = torch.zeros(M - 2, N - 2, device=self.p.device, dtype=self.p.dtype)
        BLOCK_SIZE = 1024
        pflat_n = triton.cdiv(delta.numel(), BLOCK_SIZE)
        work = torch.zeros(pflat_n, device=self.p.device, dtype=self.p.dtype)
        er = float('inf')

        while er > 1e-6:
            calc_p_right[grid](self.p, self.u, self.v, self.ub, self.vb, self.ld, p_new,
                self.p.stride(0), self.p.stride(1),
                self.u.stride(0), self.u.stride(1), self.v.stride(0), self.v.stride(1),
                self.ub.stride(0), self.ub.stride(1), self.vb.stride(0), self.vb.stride(1),
                self.ld.stride(0), self.ld.stride(1),
                self.d, self.dt, M, N, BLOCK_M=32, BLOCK_N=32)
            delta[:, :] = p_new[1:-1, 1:-1] - self.p[1:-1, 1:-1]
            er = self.max_abs_xx(delta.view(-1), work, BLOCK_SIZE)
            self.p[1:-1, 1:-1] = p_new[1:-1, 1:-1]

    def evolve_u(self):
        M, N = self.u_rhs.shape
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        evolve_u[grid](self.u_rhs, self.u, self.v, self.p, self.vb,
            self.u_rhs.stride(0), self.u_rhs.stride(1),
            self.u.stride(0), self.u.stride(1), self.v.stride(0), self.v.stride(1),
            self.p.stride(0), self.p.stride(1), self.vb.stride(0), self.vb.stride(1),
            self.d, self.dt, self.re, M, N, BLOCK_M=32, BLOCK_N=32)

    def evolve_v(self):
        M, N = self.v_rhs.shape
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        evolve_v[grid](self.v_rhs, self.u, self.v, self.p, self.ub,
            self.v_rhs.stride(0), self.v_rhs.stride(1),
            self.u.stride(0), self.u.stride(1), self.v.stride(0), self.v.stride(1),
            self.p.stride(0), self.p.stride(1), self.ub.stride(0), self.ub.stride(1),
            self.d, self.dt, self.re, M, N, BLOCK_M=32, BLOCK_N=32)

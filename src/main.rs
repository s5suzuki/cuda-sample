#[allow(non_camel_case_types)]
#[allow(dead_code)]
#[allow(deref_nullptr)]
mod cusolver;

use std::mem::size_of;

use cuda_sys::{
    cublas::{
        cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cublasOperation_t,
        cublasOperation_t_CUBLAS_OP_N,
    },
    cudart::{
        cudaMalloc, cudaMemcpy, cudaMemcpyKind_cudaMemcpyDeviceToDevice,
        cudaMemcpyKind_cudaMemcpyDeviceToHost, cudaMemcpyKind_cudaMemcpyHostToDevice, cudaMemset,
    },
};
use cusolver::{
    cudaDataType_t::CUDA_R_64F, cusolverDnCreate, cusolverDnDestroy, cusolverDnHandle_t,
    cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
};

#[link(name = "my_kernel", kind = "static")]
extern "C" {
    fn cu_add(x: *const f64, y: *mut f64, n: i32);
}

macro_rules! alloc {
    ($ty:ty, $r:expr, $c:expr) => {{
        let mut v: *mut $ty = std::ptr::null_mut();
        cudaMalloc(&mut v as *mut *mut $ty as _, size_of::<$ty>() * $r * $c);
        cudaMemset(v as _, 0, size_of::<$ty>() * $r * $c);
        (v, $r, $c)
    }};
}

macro_rules! free {
    ($p:expr) => {{
        cuda_sys::cudart::cudaFree($p.0 as _)
    }};
}

fn mat_mul(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    alpha: *const f64,
    a: (*mut f64, usize, usize),
    b: (*mut f64, usize, usize),
    beta: *const f64,
    c: (*mut f64, usize, usize),
) {
    unsafe {
        cuda_sys::cublas::cublasDgemm_v2(
            handle,
            transa,
            transb,
            c.1 as _,
            c.2 as _,
            if transa == cublasOperation_t_CUBLAS_OP_N {
                a.2
            } else {
                a.1
            } as _,
            alpha,
            a.0,
            a.1 as _,
            b.0,
            b.1 as _,
            beta,
            c.0,
            c.1 as _,
        );
    }
}

unsafe fn svd(
    handle: cusolverDnHandle_t,
    src: (*mut f64, usize, usize),
) -> (
    (*mut f64, usize, usize),
    (*mut f64, usize, usize),
    (*mut f64, usize, usize),
) {
    let m = src.1;
    let n = src.2;

    let s_size = m.min(n);

    let u = alloc!(f64, m, m);
    let s = alloc!(f64, s_size, 1);
    let vt = alloc!(f64, n, n);

    let lda = m;
    let ldu = m;
    let ldv = n;

    let mut workspace_in_bytes_on_device: u64 = 0;
    let mut workspace_in_bytes_on_host: u64 = 0;
    cusolver::cusolverDnXgesvdp_bufferSize(
        handle,
        std::ptr::null_mut(),
        CUSOLVER_EIG_MODE_VECTOR,
        0,
        m as _,
        n as _,
        CUDA_R_64F,
        src.0 as _,
        lda as _,
        CUDA_R_64F,
        s.0 as _,
        CUDA_R_64F,
        u.0 as _,
        ldu as _,
        CUDA_R_64F,
        vt.0 as _,
        ldv as _,
        CUDA_R_64F,
        &mut workspace_in_bytes_on_device as _,
        &mut workspace_in_bytes_on_host as _,
    );

    let workspace_buffer_on_device = alloc!(u8, workspace_in_bytes_on_device as usize, 1);
    let mut workspace_buffer_on_host_v = vec![0u8; workspace_in_bytes_on_host as usize];
    let workspace_buffer_on_host = if workspace_in_bytes_on_host > 0 {
        workspace_buffer_on_host_v.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };

    let info = alloc!(i32, 1, 1);

    let mut h_err_sigma = 0.;
    cusolver::cusolverDnXgesvdp(
        handle,
        std::ptr::null_mut(),
        CUSOLVER_EIG_MODE_VECTOR,
        0,
        m as _,
        n as _,
        CUDA_R_64F,
        src.0 as _,
        lda as _,
        CUDA_R_64F,
        s.0 as _,
        CUDA_R_64F,
        u.0 as _,
        ldu as _,
        CUDA_R_64F,
        vt.0 as _,
        ldv as _,
        CUDA_R_64F,
        workspace_buffer_on_device.0 as _,
        workspace_in_bytes_on_device,
        workspace_buffer_on_host as _,
        workspace_in_bytes_on_host,
        info.0 as _,
        &mut h_err_sigma as _,
    );

    free!(info);
    free!(workspace_buffer_on_device);

    (u, s, vt)
}

fn main() {
    unsafe {
        let m = 2;
        let n = 1024;

        let one = 1.0;
        let zero = 0.0;
        let a_ = vec![one; m * n];
        let b_ = vec![one; m * n];

        let a = alloc!(f64, m, n);
        let b = alloc!(f64, n, m);
        let c = alloc!(f64, m, m);
        cudaMemcpy(
            a.0 as _,
            a_.as_ptr() as _,
            m * n * size_of::<f64>(),
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        );
        cudaMemcpy(
            b.0 as _,
            b_.as_ptr() as _,
            m * n * size_of::<f64>(),
            cudaMemcpyKind_cudaMemcpyHostToDevice,
        );

        let mut handle: cublasHandle_t = std::ptr::null_mut();
        cublasCreate_v2(&mut handle as *mut _);

        mat_mul(
            handle,
            cublasOperation_t_CUBLAS_OP_N,
            cublasOperation_t_CUBLAS_OP_N,
            &one,
            a,
            b,
            &zero,
            c,
        );

        cu_add(a.0, c.0, (m * m) as _);

        let mut handle_s: cusolverDnHandle_t = std::ptr::null_mut();
        cusolverDnCreate(&mut handle_s as *mut _);

        let (u, s, vt) = svd(handle_s, c);

        let sm = alloc!(f64, m, m);
        cudaMemcpy(
            sm.0 as _,
            s.0 as _,
            size_of::<f64>(),
            cudaMemcpyKind_cudaMemcpyDeviceToDevice,
        );
        cudaMemcpy(
            sm.0.add(2) as _,
            s.0.add(1) as _,
            size_of::<f64>(),
            cudaMemcpyKind_cudaMemcpyDeviceToDevice,
        );

        let tmp = alloc!(f64, m, m);
        mat_mul(
            handle,
            cublasOperation_t_CUBLAS_OP_N,
            cublasOperation_t_CUBLAS_OP_N,
            &one,
            u,
            sm,
            &zero,
            tmp,
        );
        mat_mul(
            handle,
            cublasOperation_t_CUBLAS_OP_N,
            cublasOperation_t_CUBLAS_OP_N,
            &one,
            tmp,
            vt,
            &zero,
            c,
        );

        cublasDestroy_v2(handle);
        cusolverDnDestroy(handle_s);

        let mut c_: Vec<f64> = vec![zero; m * m];
        cudaMemcpy(
            c_.as_mut_ptr() as _,
            c.0 as _,
            c_.len() * size_of::<f64>(),
            cudaMemcpyKind_cudaMemcpyDeviceToHost,
        );
        println!("{:?}", c_);

        free!(a);
        free!(b);
        free!(c);
        free!(u);
        free!(s);
        free!(vt);
    }
}

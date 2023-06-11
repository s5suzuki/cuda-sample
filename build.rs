/*
 * File: build.rs
 * Project: src
 * Created Date: 11/06/2023
 * Author: Shun Suzuki
 * -----
 * Last Modified: 11/06/2023
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2023 Shun Suzuki. All rights reserved.
 *
 */

use cuda_config::*;

fn main() {
    println!("cargo:rerun-if-changed=kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");

    cc::Build::new()
        .cuda(true)
        .flag("-allow-unsupported-compiler")
        .flag("-cudart=shared")
        .flag("-gencode=arch=compute_75,code=sm_75")
        .flag("-gencode=arch=compute_80,code=sm_80")
        .flag("-gencode=arch=compute_86,code=sm_86")
        .flag("-gencode=arch=compute_87,code=sm_87")
        .file("kernel.cu")
        .compile("my_kernel");

    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}",
            find_cuda_windows().display()
        );
    } else {
        for path in find_cuda() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    };

    println!("cargo:rustc-link-lib=dylib=cusolver");
}

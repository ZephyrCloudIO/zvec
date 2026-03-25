use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir.join("../.."); // zvec root

    // Header path
    let header = project_root.join("src/include/zvec/c_api.h");

    // Link against zvec_c_api shared library
    // On Windows MSVC builds, libs are under build/lib/{Release,Debug}
    let build_profile = env::var("ZVEC_BUILD_PROFILE").unwrap_or_else(|_| "Release".to_string());
    let lib_dir = project_root.join("build/lib").join(&build_profile);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // Also search the flat build/lib dir (Linux/macOS)
    println!(
        "cargo:rustc-link-search=native={}",
        project_root.join("build/lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=zvec_c_api");

    // Rebuild if header changes
    println!("cargo:rerun-if-changed={}", header.display());

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .clang_arg(format!(
            "-I{}",
            project_root.join("src/include").display()
        ))
        .allowlist_function("zvec_.*")
        .allowlist_type("ZVec.*")
        .allowlist_var("ZVEC_.*")
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ffi.rs"))
        .expect("Failed to write bindings");
}

fn main() {
    prost_build::Config::new()
        .out_dir(std::env::var("OUT_DIR").expect("OUT_DIR env var"))
        .compile_protos(&["proto/dex.proto"], &["proto"])
        .expect("compile protos");
}

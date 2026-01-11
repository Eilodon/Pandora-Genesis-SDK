fn main() {
    // Build script context: if codegen fails, fail the build with a clear message.
    if let Err(err) =
        prost_build::compile_protos(&["proto/ontology.proto", "proto/core.proto"], &["proto/"])
    {
        println!("cargo:warning=Failed to compile protobufs: {}", err);
        println!("cargo:rerun-if-changed=proto/ontology.proto");
        println!("cargo:rerun-if-changed=proto/core.proto");
        // In build.rs, returning error is not supported; explicitly exit with failure.
        std::process::exit(1);
    }
    println!("cargo:rerun-if-changed=proto/ontology.proto");
    println!("cargo:rerun-if-changed=proto/core.proto");
}

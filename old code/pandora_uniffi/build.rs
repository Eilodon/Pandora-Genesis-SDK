fn main() {
    if let Err(err) = uniffi::generate_scaffolding("./src/pandora.udl") {
        println!(
            "cargo:warning=Failed to generate uniffi scaffolding: {}",
            err
        );
        println!("cargo:rerun-if-changed=src/pandora.udl");
        std::process::exit(1);
    }
    println!("cargo:rerun-if-changed=src/pandora.udl");
}

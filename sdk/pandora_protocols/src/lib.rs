// Tên file proto là ontology.proto -> prost sẽ tạo ra file b_one.protocols.rs
pub mod b_one {
    pub mod protocols {
        include!(concat!(env!("OUT_DIR"), "/b_one.protocols.rs"));
    }
    pub mod core {
        include!(concat!(env!("OUT_DIR"), "/b_one.core.rs"));
    }
}

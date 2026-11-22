//! Script run at compile-time. Compiles tonic gRPC protos.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("../../../proto/tensorio.proto")?;
    Ok(())
}

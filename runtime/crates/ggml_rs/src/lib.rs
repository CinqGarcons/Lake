//! The ggml_rs crate contains Lake's Rust bindings for ggml

/// Hello world placeholder.
pub fn hello() {
    println!("Hello from GGML bindings!")
}

/// Placeholder Tensor struct
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Tensor {
    /// Placeholder data
    pub data: Vec<u8>,
    /// Placeholder ne
    pub ne: [i64; 4],
    /// Placeholder nb
    pub nb: [u64; 4],
    /// Placeholder dtype
    pub dtype: u32,
}

impl Tensor {
    /// Placeholder method for copying a tensor out of GGML
    #[inline]
    pub fn read_all(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Placeholder method for getting GGML tensor ne metadata
    #[inline]
    pub fn ne(&self) -> [i64; 4] {
        self.ne
    }

    /// Placeholder method for getting GGML tensor nb metadata
    #[inline]
    pub fn nb(&self) -> [u64; 4] {
        self.nb
    }

    /// Placeholder method for getting GGML tensor dtype metadata
    #[inline]
    pub fn dtype(&self) -> u32 {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        assert_eq!(1, 1);
    }
}

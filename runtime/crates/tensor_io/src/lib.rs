//! The tensor_io crate contains the high-performance logic for sending tensors between Lake nodes.
//!
//! Currently, the following Tensor IO implementations are available:
//! - `tensor_io::basic::TensorIO`
//!
//! TensorIO implementations follow the [TypeState](<https://cliffle.com/blog/rust-typestate/>) pattern for initialization.
//! This prevents (at compile-time) users from calling `send()` or `recv()` if the client or server
//! (respectively) haven't been set up yet (via `connect()` and `serve()`).
//!
//! Contrived example of a TensorIO connecting to itself.
//! Explicit (but unnecessary) typing parameters are used to help demonstrate how TypeState works:
//! ```rust
//! use std::net::SocketAddr;
//!
//! use ggml_rs::Tensor;
//! use tensor_io::basic::{Connected, Disconnected, Serving, TensorIO};
//! use tensor_io::{Connect, Disconnect, Receiver, Sender, Serve, Step, StopServe};
//!
//! let step_in = Step {
//!     activations: Tensor {
//!         data: vec![1, 2, 3],
//!         ne: [1, 2, 3, 4],
//!         nb: [1, 2, 3, 4],
//!         dtype: 3,
//!     },
//!     request_id: "test_request".to_owned(),
//!     temperature: 1.5,
//! };
//!
//! // Initialize a new basic::TensorIO
//! let tio: TensorIO<Disconnected, Disconnected> = TensorIO::new();
//!
//! // Start the TensorIO server on a port of the OS's choosing.
//! // Returns the updated TensorIO object, and the address it bound to.
//! let (tio, addr) = tio.serve(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
//!
//! // Start the TensorIO client, connected to the server above.
//! // For this example, we're connecting a single TensorIO to itself, but you'd probably
//! // want to connect multiple different ones to each other.
//! //
//! // Also notice how the type has changed from TensorIO<Disconnected, Disconnected> at first,
//! // to TensorIO<Serving, Connected> now. This is TypeState in action.
//! let mut tio: TensorIO<Serving, Connected> = tio.connect(addr).unwrap();
//!
//! // Send the step into the sender, then take it out of the receiver.
//! //
//! // Now that our tio is TensorIO<Serving, Connected>, both the send() and recv()
//! // methods are available to us.
//! tio.send(step_in.clone()).unwrap();
//! let step_out = tio.recv().unwrap();
//!
//! assert_eq!(step_in, step_out);
//!
//! // Cleanup: close the connections and stop the server.
//! let tio: TensorIO<Serving, Disconnected> = tio.disconnect();
//! let tio: TensorIO<Disconnected, Disconnected> = tio.stop_srv();
//! ```
use std::net::SocketAddr;

use anyhow::Result;
use ggml_rs::Tensor;

pub mod basic;

/// TensorIO trait is a convenience alias for Sender + Receiver
/// Implemented by a fully-connected TensorIO implementation
pub trait TensorIO: Sender + Receiver {}
impl<T: Sender + Receiver> TensorIO for T {}

/// A single inference step.
#[derive(PartialEq, Clone, Debug)]
pub struct Step {
    /// ID associated with the source inference request.
    pub request_id: String,
    /// Tensor containing the resulting activations from the step.
    pub activations: Tensor,
    /// The temperature to use for inference in this step.
    pub temperature: f64,
}

/// A Tensor IO sender.
pub trait Sender {
    /// Send a step to the downstream node.
    /// May block until the downstream node is ready.
    fn send(&self, step: Step) -> Result<()>;
}

/// A TensorIO receiver.
pub trait Receiver {
    /// Receive a step from the upstream node.
    /// May block until a step from the upstream node is available.
    fn recv(&mut self) -> Option<Step>;
}

/// Connect the TensorIO client to a receiver server.
pub trait Connect {
    /// Success return type (TypeState)
    type Output;
    /// Start the TensorIO client to connect to a receiver server.
    fn connect(self, downstream: SocketAddr) -> Result<Self::Output>;
}

/// Start the TensorIO server.
pub trait Serve {
    /// Success return type (TypeState)
    type Output;
    /// Start the TensorIO server.
    ///
    /// Serves on address serve_on.
    ///
    /// Returns a result with the new TensorIO struct, and the resulting address
    /// it was bound to.
    fn serve(self, serve_on: SocketAddr) -> Result<(Self::Output, SocketAddr)>;
}

/// Disconnect the TensorIO client from its receiver server.
pub trait Disconnect {
    /// Return type (TypeState)
    type Output;
    /// Disconnect the TensorIO client from its receiver server.
    fn disconnect(self) -> Self::Output;
}

/// Shut down the TensorIO server.
pub trait StopServe {
    /// Return type (TypeState)
    type Output;
    /// Shut down the TensorIO server
    fn stop_srv(self) -> Self::Output;
}

//! Minimal tensor_io::IO implementation. No compression.
use std::net::SocketAddr;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::Result;
use ggml_rs::Tensor;
use tokio::runtime::Builder;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::transport::{Endpoint, Server};
use tonic::{Request, Response, Status, Streaming};

use crate::{Connect, Disconnect, Receiver, Sender, Serve, Step, StopServe};

mod proto {
    tonic::include_proto!("tensorio");
}

use proto::tio_service_client::TioServiceClient;
use proto::tio_service_server::{TioService, TioServiceServer};
use proto::{Ack, TioStep};

const MAX_MESSAGE_SIZE: usize = 3 * 1024 * 1024 * 1024;

/// Disconnected state (<https://cliffle.com/blog/rust-typestate/>)
pub struct Disconnected;

/// Serving state (<https://cliffle.com/blog/rust-typestate/>)
pub struct Serving {
    thread: JoinHandle<Result<()>>,
    rx: mpsc::Receiver<TioStep>,
    shutdown: oneshot::Sender<()>,
}

impl Serving {
    fn shutdown(self) {
        drop(self.rx);
        drop(self.shutdown);
        let _ = self.thread.join();
    }
}

/// Connected state (<https://cliffle.com/blog/rust-typestate/>)
pub struct Connected {
    thread: JoinHandle<Result<()>>,
    tx: mpsc::Sender<TioStep>,
}

impl Connected {
    fn shutdown(self) {
        drop(self.tx);
        let _ = self.thread.join();
    }
}

/// TypeState trait (<https://cliffle.com/blog/rust-typestate/>)
/// Represents the state of the TensorIO server (Disconnected, Serving)
pub trait SrvState {}
impl SrvState for Disconnected {}
impl SrvState for Serving {}

/// TypeState trait (<https://cliffle.com/blog/rust-typestate/>)
/// Represents the state of the TensorIO client (Disconnected, Connected)
pub trait CliState {}
impl CliState for Disconnected {}
impl CliState for Connected {}

/// Minimal tensor_io::Sender and tensor_io::Receiver implementation. No compression.
pub struct TensorIO<S = Disconnected, C = Disconnected>
where
    S: SrvState,
    C: CliState,
{
    srv: S,
    cli: C,
}

impl TensorIO<Disconnected> {
    /// Create a new TensorIO
    pub fn new() -> Self {
        Self {
            srv: Disconnected {},
            cli: Disconnected {},
        }
    }
}

impl Default for TensorIO<Disconnected> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: SrvState> Connect for TensorIO<S, Disconnected> {
    type Output = TensorIO<S, Connected>;

    fn connect(self, downstream: SocketAddr) -> Result<Self::Output> {
        let (connection, ready) = start_client(downstream);

        if let Err(e) = ready.blocking_recv()? {
            connection.shutdown();
            return Err(e);
        }

        Ok(TensorIO {
            srv: self.srv,
            cli: connection,
        })
    }
}

impl<C: CliState> Serve for TensorIO<Disconnected, C> {
    type Output = TensorIO<Serving, C>;

    fn serve(self, serve_on: SocketAddr) -> Result<(Self::Output, SocketAddr)> {
        let (server, ready) = start_server(serve_on);

        let addr = match ready.blocking_recv()? {
            Ok(a) => a,
            Err(e) => {
                server.shutdown();
                return Err(e);
            }
        };

        Ok((
            TensorIO {
                srv: server,
                cli: self.cli,
            },
            addr,
        ))
    }
}

impl<S: SrvState> Disconnect for TensorIO<S, Connected> {
    type Output = TensorIO<S, Disconnected>;

    fn disconnect(self) -> Self::Output {
        self.cli.shutdown();

        TensorIO {
            srv: self.srv,
            cli: Disconnected {},
        }
    }
}

impl<C: CliState> StopServe for TensorIO<Serving, C> {
    type Output = TensorIO<Disconnected, C>;

    fn stop_srv(self) -> Self::Output {
        self.srv.shutdown();

        TensorIO {
            srv: Disconnected {},
            cli: self.cli,
        }
    }
}

impl<S: SrvState> Sender for TensorIO<S, Connected> {
    fn send(&self, step: Step) -> Result<()> {
        self.cli.tx.blocking_send(TioStep {
            ne_0: step.activations.ne()[0],
            ne_1: step.activations.ne()[1],
            ne_2: step.activations.ne()[2],
            ne_3: step.activations.ne()[3],
            nb_0: step.activations.nb()[0],
            nb_1: step.activations.nb()[1],
            nb_2: step.activations.nb()[2],
            nb_3: step.activations.nb()[3],
            dtype: step.activations.dtype(),
            request_id: step.request_id,
            temperature: step.temperature,
            data: step.activations.read_all(),
        })?;

        Ok(())
    }
}

impl<C: CliState> Receiver for TensorIO<Serving, C> {
    fn recv(&mut self) -> Option<Step> {
        let step = self.srv.rx.blocking_recv()?;

        Some(Step {
            activations: Tensor {
                ne: [step.ne_0, step.ne_1, step.ne_2, step.ne_3],
                nb: [step.nb_0, step.nb_1, step.nb_2, step.nb_3],
                data: step.data,
                dtype: step.dtype,
            },
            request_id: step.request_id,
            temperature: step.temperature,
        })
    }
}

fn start_client(downstream: SocketAddr) -> (Connected, oneshot::Receiver<Result<()>>) {
    let (tx, rx) = mpsc::channel::<TioStep>(1);
    let (ready_tx, ready_rx) = oneshot::channel::<Result<()>>();

    let thread = thread::spawn(move || -> Result<()> {
        let runtime = Builder::new_multi_thread().enable_all().build()?;
        runtime.block_on(client_thread(downstream, rx, ready_tx))?;
        Ok(())
    });

    (Connected { thread, tx }, ready_rx)
}

async fn client_thread(
    downstream: SocketAddr,
    rx: mpsc::Receiver<TioStep>,
    ready_tx: oneshot::Sender<Result<()>>,
) -> Result<()> {
    let uri = format!("http://{downstream}");
    let endpoint = match Endpoint::from_shared(uri) {
        Ok(ep) => ep,
        Err(e) => {
            let _ = ready_tx.send(Err(anyhow::Error::from(e).context("Invalid downstream URI")));
            return Err(anyhow::anyhow!("Invalid downstream URI"));
        }
    };
    let endpoint = endpoint
        .connect_timeout(Duration::from_secs(10))
        .keep_alive_while_idle(true);

    let channel = match endpoint.connect().await {
        Ok(ch) => ch,
        Err(e) => {
            let _ = ready_tx.send(Err(
                anyhow::Error::from(e).context("gRPC connection failure")
            ));
            return Err(anyhow::anyhow!("gRPC connection failure"));
        }
    };

    let _ = ready_tx.send(Ok(()));

    let mut client = TioServiceClient::new(channel).max_encoding_message_size(MAX_MESSAGE_SIZE);
    let stream = ReceiverStream::new(rx);

    // TODO: Actually propagate error to sender instead of storing it in the JoinHandle
    client.send(Request::new(stream)).await?;
    Ok(())
}

fn start_server(serve_on: SocketAddr) -> (Serving, oneshot::Receiver<Result<SocketAddr>>) {
    let (tx, rx) = mpsc::channel(1);
    let (ready_tx, ready_rx) = oneshot::channel();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let thread = thread::spawn(move || -> Result<()> {
        let runtime = Builder::new_multi_thread().enable_all().build()?;
        runtime.block_on(server_thread(serve_on, tx, ready_tx, shutdown_rx))?;
        Ok(())
    });

    (
        Serving {
            thread,
            rx,
            shutdown: shutdown_tx,
        },
        ready_rx,
    )
}

async fn server_thread(
    serve_on: SocketAddr,
    tx: mpsc::Sender<TioStep>,
    ready_tx: oneshot::Sender<Result<SocketAddr>>,
    shutdown: oneshot::Receiver<()>,
) -> Result<()> {
    let service = TIOServer { tx };
    let svc = TioServiceServer::new(service).max_decoding_message_size(MAX_MESSAGE_SIZE);
    let server = Server::builder().add_service(svc);

    let listener = match tokio::net::TcpListener::bind(serve_on).await {
        Ok(l) => l,
        Err(e) => {
            let _ = ready_tx.send(Err(
                anyhow::Error::from(e).context("Error binding to given address")
            ));
            return Err(anyhow::anyhow!("Error binding to given address"));
        }
    };
    let addr = match listener.local_addr() {
        Ok(a) => a,
        Err(e) => {
            let _ = ready_tx.send(Err(
                anyhow::Error::from(e).context("Error getting local address")
            ));
            return Err(anyhow::anyhow!("Error getting local address"));
        }
    };

    let _ = ready_tx.send(Ok(addr));

    server
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
            let _ = shutdown.await;
        })
        .await?;

    Ok(())
}

struct TIOServer {
    tx: mpsc::Sender<TioStep>,
}

#[tonic::async_trait]
impl TioService for TIOServer {
    async fn send(
        &self,
        request: Request<Streaming<TioStep>>,
    ) -> std::result::Result<Response<Ack>, Status> {
        let mut stream = request.into_inner();

        while let Some(tensor) = stream.message().await? {
            if let Err(e) = self.tx.send(tensor).await {
                return Err(Status::internal(format!("Error receiving tensor: {}", e)));
            };
        }

        Ok(Response::new(Ack {}))
    }
}

#[cfg(test)]
mod tests {
    use rstest::*;

    use super::*;

    #[fixture]
    fn tensor_io() -> TensorIO {
        TensorIO::new()
    }

    #[fixture]
    fn bidirectional_tio(
        #[from(tensor_io)] tio_a: TensorIO,
        #[from(tensor_io)] tio_b: TensorIO,
    ) -> (TensorIO<Serving, Connected>, TensorIO<Serving, Connected>) {
        let (tio_a, addr_a) = tio_a.serve(([127, 0, 0, 1], 0).into()).unwrap();
        let (tio_b, addr_b) = tio_b.serve(([127, 0, 0, 1], 0).into()).unwrap();

        let tio_a = tio_a.connect(addr_b).unwrap();
        let tio_b = tio_b.connect(addr_a).unwrap();

        (tio_a, tio_b)
    }

    #[fixture]
    fn dummy_tensor() -> Tensor {
        Tensor {
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ne: [2, 3, 4, 5],
            nb: [3, 4, 5, 6],
            dtype: 4,
        }
    }

    #[fixture]
    fn dummy_step(dummy_tensor: Tensor) -> Step {
        Step {
            activations: dummy_tensor,
            request_id: "test_request".to_owned(),
            temperature: 123.,
        }
    }

    #[rstest]
    #[should_panic(expected = "gRPC connection failure")]
    fn connect_to_unused_port(tensor_io: TensorIO) {
        tensor_io.connect(([127, 0, 0, 1], 9999).into()).unwrap();
    }

    #[rstest]
    #[should_panic(expected = "gRPC connection failure")]
    fn connect_to_bad_addr(tensor_io: TensorIO) {
        tensor_io
            .connect(([255, 255, 255, 255], 9999).into())
            .unwrap();
    }

    #[rstest]
    fn connects_unidirectional(
        #[from(tensor_io)] tio_a: TensorIO,
        #[from(tensor_io)] tio_b: TensorIO,
    ) {
        let (tio_a, addr) = tio_a.serve(([127, 0, 0, 1], 0).into()).unwrap();

        let tio_b = tio_b.connect(addr).unwrap();

        tio_b.disconnect();
        tio_a.stop_srv();
    }

    #[rstest]
    fn connects_bidirectional(
        #[from(tensor_io)] tio_a: TensorIO,
        #[from(tensor_io)] tio_b: TensorIO,
    ) {
        let (tio_a, addr_a) = tio_a.serve(([127, 0, 0, 1], 0).into()).unwrap();
        let (tio_b, addr_b) = tio_b.serve(([127, 0, 0, 1], 0).into()).unwrap();

        let tio_a = tio_a.connect(addr_b).unwrap();
        let tio_b = tio_b.connect(addr_a).unwrap();

        let tio_a = tio_a.disconnect();
        let tio_b = tio_b.disconnect();
        tio_a.stop_srv();
        tio_b.stop_srv();
    }

    #[rstest]
    fn connects_to_self(#[from(tensor_io)] tio: TensorIO) {
        let (tio, addr) = tio.serve(([127, 0, 0, 1], 0).into()).unwrap();

        let tio = tio.connect(addr).unwrap();

        let tio = tio.disconnect();
        tio.stop_srv();
    }

    #[rstest]
    fn sends_tensors_bidirectional(
        bidirectional_tio: (TensorIO<Serving, Connected>, TensorIO<Serving, Connected>),
        #[from(dummy_step)] ia: Step,
        #[from(dummy_step)] ib: Step,
    ) {
        let (mut tio_a, mut tio_b) = bidirectional_tio;

        tio_a.send(ia.clone()).unwrap();
        tio_b.send(ib.clone()).unwrap();
        let oa = tio_b.recv().unwrap();
        let ob = tio_a.recv().unwrap();

        assert_eq!(ia, oa);
        assert_eq!(ib, ob);

        let tio_a = tio_a.disconnect();
        let tio_b = tio_b.disconnect();
        tio_a.stop_srv();
        tio_b.stop_srv();
    }

    #[rstest]
    fn sends_tensors_to_self(#[from(tensor_io)] tio: TensorIO, dummy_step: Step) {
        let (tio, addr) = tio.serve(([127, 0, 0, 1], 0).into()).unwrap();
        let mut tio = tio.connect(addr).unwrap();

        tio.send(dummy_step.clone()).unwrap();
        let output_step = tio.recv().unwrap();

        assert_eq!(dummy_step, output_step);

        let tio = tio.disconnect();
        tio.stop_srv();
    }
}

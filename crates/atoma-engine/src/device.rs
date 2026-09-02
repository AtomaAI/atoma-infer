//! What a rank's device holds, allocated in the Allocation session phase: the device candle
//! computes on, the KV cache, the model weights and, under NCCL, the communicator.
//!
//! Every constructor here takes the [`Allocation`] phase as a witness: the session fixes device
//! addresses before anything is captured, and taking the phase by reference is what makes an
//! allocation after that point unwritable. In this crate the forward runs eagerly on candle's own
//! stream, so nothing is enqueued through the session's descriptor seam; the session orders
//! startup and nothing else.

use std::path::Path;
use std::sync::Arc;

use atoma_runtime::session::Allocation;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cudarc::driver::CudaStream;
use models::llama::{Config, LlamaConfig};
use models::FlashAttention;
use thiserror::Error;
use tracing::info;

use crate::config::{DeviceOrdinal, Dtype};
use crate::model::{read_json, ModelError, ModelFiles};

#[cfg(feature = "nccl")]
use std::rc::Rc;

#[cfg(feature = "nccl")]
use cudarc::nccl::{Comm, Id};

#[cfg(feature = "nccl")]
use crate::config::Rank;

/// Why a rank's device could not be set up.
#[derive(Debug, Error)]
pub enum DeviceError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[cfg(feature = "nccl")]
    #[error(
        "the NCCL communicator for rank {rank} of {world_size} could not be opened: {status:?}"
    )]
    Nccl {
        rank: Rank,
        world_size: usize,
        status: cudarc::nccl::sys::ncclResult_t,
    },
}

impl From<Dtype> for DType {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F16 => Self::F16,
            Dtype::Bf16 => Self::BF16,
            Dtype::F32 => Self::F32,
        }
    }
}

/// The CUDA device a rank computes on, and the stream candle runs it on.
pub struct RankDevice {
    device: Device,
    stream: Arc<CudaStream>,
}

impl RankDevice {
    /// Opens device `ordinal` for candle.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Candle`] when candle cannot open the device.
    pub fn open(_allocation: &Allocation, ordinal: DeviceOrdinal) -> Result<Self, DeviceError> {
        let device = Device::new_cuda(ordinal.get())?;
        let stream = device.as_cuda_device()?.cuda_stream();
        info!(ordinal = ordinal.get(), "device opened");
        Ok(Self { device, stream })
    }

    #[must_use]
    pub fn candle(&self) -> &Device {
        &self.device
    }

    /// The stream candle computes on: where the forward's work is enqueued, and so where its
    /// readback must be.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// The model's configuration as its `config.json` declares it.
///
/// # Errors
///
/// Returns [`DeviceError::Model`] when the file cannot be read or is not a Llama configuration.
pub fn read_config(config: &Path) -> Result<Config, DeviceError> {
    let declared: LlamaConfig = read_json(config)?;
    Ok(declared.into_config())
}

/// The KV cache's shape for one rank: how many blocks, of how many tokens, over how many of the
/// model's key-value heads this rank holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeometry {
    pub block_count: usize,
    pub block_size: usize,
    /// The key-value heads this rank holds: all of them alone, a share of them under tensor
    /// parallelism.
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl KvGeometry {
    /// The geometry of `config`'s cache over `block_count` blocks of `block_size`, with the
    /// key-value heads split `world_size` ways.
    #[must_use]
    pub fn new(config: &Config, block_count: usize, block_size: usize, world_size: usize) -> Self {
        Self {
            block_count,
            block_size,
            kv_heads: config.num_key_value_heads / world_size,
            head_dim: config.hidden_size / config.num_attention_heads,
        }
    }
}

/// One rank's paged KV cache: one tensor per layer, every block's keys and values.
pub struct KvCache {
    layers: Vec<Tensor>,
}

impl KvCache {
    /// Allocates `config.num_hidden_layers` zeroed layers of `geometry` in `dtype` on `device`.
    /// The block count is allocated as configured; a device that cannot hold it fails here.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Candle`] when the device cannot hold the cache.
    pub fn allocate(
        _allocation: &Allocation,
        device: &RankDevice,
        config: &Config,
        geometry: KvGeometry,
        dtype: DType,
    ) -> Result<Self, DeviceError> {
        let shape = FlashAttention::get_kv_cache_shape(
            geometry.block_count,
            geometry.block_size,
            geometry.kv_heads,
            geometry.head_dim,
        );
        let layers = (0..config.num_hidden_layers)
            .map(|_| Tensor::zeros(shape.clone(), dtype, device.candle()))
            .collect::<Result<Vec<_>, _>>()?;
        info!(
            layers = layers.len(),
            blocks = geometry.block_count,
            block_size = geometry.block_size,
            kv_heads = geometry.kv_heads,
            head_dim = geometry.head_dim,
            "KV cache allocated"
        );
        Ok(Self { layers })
    }

    /// Every layer's cache, in layer order, for the forward to write.
    pub fn layers_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut().collect()
    }
}

/// The Llama the forward runs: over all its heads alone, or over this rank's share of them
/// under NCCL.
#[cfg(not(feature = "nccl"))]
pub type Llama = models::llama::Llama;
#[cfg(feature = "nccl")]
pub type Llama = models::llama_nccl::Llama;

/// The NCCL communicator binding this rank's stream into the collective.
#[cfg(feature = "nccl")]
pub struct Communicator {
    comm: Rc<Comm>,
}

#[cfg(feature = "nccl")]
impl Communicator {
    /// Joins the collective `id` as `rank` of `world_size`, ordered against `device`'s stream so
    /// the collectives run where the model computes. Returns once every rank has joined.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Nccl`] when NCCL refuses the rank.
    pub fn open(
        _allocation: &Allocation,
        device: &RankDevice,
        rank: Rank,
        world_size: usize,
        id: Id,
    ) -> Result<Self, DeviceError> {
        let comm = Comm::from_rank(Arc::clone(device.stream()), rank.get(), world_size, id)
            .map_err(|error| DeviceError::Nccl {
                rank,
                world_size,
                status: error.0,
            })?;
        info!(%rank, world_size, "NCCL communicator opened");
        Ok(Self {
            comm: Rc::new(comm),
        })
    }
}

/// What the weights are loaded from: the fetched files, the configuration they were saved
/// under, and the dtype they are loaded in.
#[derive(Debug, Clone, Copy)]
pub struct Checkpoint<'a> {
    pub files: &'a ModelFiles,
    pub config: &'a Config,
    pub dtype: DType,
}

/// The model's weights, loaded onto one rank's device.
pub struct Weights {
    llama: Llama,
}

impl Weights {
    /// Memory-maps the checkpoint's shards and loads them onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Candle`] when the shards cannot be mapped or loaded.
    #[cfg(not(feature = "nccl"))]
    pub fn load(
        _allocation: &Allocation,
        device: &RankDevice,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Self, DeviceError> {
        let vb = Self::var_builder(device, checkpoint)?;
        let llama = Llama::load(vb, checkpoint.config, checkpoint.dtype, device.candle())?;
        info!(shards = checkpoint.files.weights.len(), "weights loaded");
        Ok(Self { llama })
    }

    /// Memory-maps the checkpoint's shards and loads this rank's share of them onto `device`,
    /// over `communicator`.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Candle`] when the shards cannot be mapped or loaded, or the model
    /// cannot be split across the communicator's ranks.
    #[cfg(feature = "nccl")]
    pub fn load(
        _allocation: &Allocation,
        device: &RankDevice,
        checkpoint: Checkpoint<'_>,
        communicator: &Communicator,
    ) -> Result<Self, DeviceError> {
        let vb = Self::var_builder(device, checkpoint)?;
        let llama = Llama::load(
            vb,
            checkpoint.config,
            &communicator.comm,
            checkpoint.dtype,
            device.candle(),
        )?;
        info!(shards = checkpoint.files.weights.len(), "weights loaded");
        Ok(Self { llama })
    }

    fn var_builder(
        device: &RankDevice,
        checkpoint: Checkpoint<'_>,
    ) -> Result<VarBuilder<'static>, DeviceError> {
        // SAFETY: the shards are memory-mapped and must not change underneath the map; they are
        // files in the Hub cache, written once when fetched.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &checkpoint.files.weights,
                checkpoint.dtype,
                device.candle(),
            )
        }?;
        Ok(vb)
    }

    pub fn llama_mut(&mut self) -> &mut Llama {
        &mut self.llama
    }
}

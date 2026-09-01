//! GPU-free engine core.
//!
//! Hosts the decisions the engine makes on the host side: which captured CUDA graph serves a live
//! batch, what an attention backend must declare and expose to be captured at all, and the shared
//! id and count types those decisions are written in. The crate links no driver, and every test
//! runs without a GPU.

pub mod attention;
pub mod config;
pub mod dispatch;
pub mod engine;
pub mod kv;
pub mod request;
pub mod scheduler;
pub mod step;
pub mod types;

#[cfg(test)]
pub(crate) mod test_support {
    use std::io::Write;
    use std::sync::{Arc, Mutex, OnceLock};

    use tracing::subscriber::set_global_default;
    use tracing::Level;

    use crate::attention::{
        BackendDeclaration, BreakSite, CaptureContract, ModelDeclaration, SupportLevel,
    };
    use crate::types::{RequestCount, TokenCount};

    pub(crate) fn tokens(value: usize) -> TokenCount {
        TokenCount::new(value).expect("test token counts are nonzero")
    }

    pub(crate) fn requests(value: usize) -> RequestCount {
        RequestCount::new(value).expect("test request counts are nonzero")
    }

    pub(crate) fn site(layer: usize, op: usize) -> BreakSite {
        BreakSite { layer, op }
    }

    /// What one backend declaring `support_level` and unable to capture `sites` settles, beside a
    /// model with no eager region of its own. What most tests outside the attention module need
    /// from a contract: a graph mode, and whether anything breaks the pass.
    pub(crate) fn contract(support_level: SupportLevel, sites: &[BreakSite]) -> CaptureContract {
        let backend = sites.iter().fold(
            BackendDeclaration::new("test-backend", support_level),
            |declaration, &site| declaration.cannot_capture(site),
        );
        CaptureContract::resolve(&[backend], &ModelDeclaration::new("test-model"))
    }

    /// Log output collected behind the process-global subscriber.
    ///
    /// A thread-scoped subscriber is unreliable here: with a single registered dispatcher,
    /// tracing computes a callsite's cached interest on whichever thread first hits it, and a
    /// sibling test's thread without a subscriber caches a no-op interest for everyone. The
    /// global subscriber sees every test's events, so assertions must match on values unique to
    /// their own test.
    #[derive(Clone, Default)]
    pub(crate) struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl CapturedLog {
        pub(crate) fn contents(&self) -> String {
            let bytes = self.0.lock().expect("log capture lock").clone();
            String::from_utf8(bytes).expect("log output is utf-8")
        }
    }

    impl Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("log capture lock")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The one log capture of this test binary. A second global subscriber cannot be installed,
    /// so every test that reads log output shares this one.
    pub(crate) fn captured_log() -> &'static CapturedLog {
        static LOG: OnceLock<CapturedLog> = OnceLock::new();
        LOG.get_or_init(|| {
            let log = CapturedLog::default();
            let writer = log.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(Level::DEBUG)
                .with_ansi(false)
                .with_writer(move || writer.clone())
                .finish();
            set_global_default(subscriber).expect("no other test installs a subscriber");
            log
        })
    }
}

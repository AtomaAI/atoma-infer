//! Reading a configuration from a TOML file and the environment.
//!
//! A TOML file supplies the whole shape and environment variables under `ATOMA_` override
//! individual fields, nesting with `__`, so a deployment can change one setting without rewriting
//! the file: `ATOMA_SCHEDULER__TOKEN_BUDGET=4096` sets `token_budget` under `[scheduler]`.
//!
//! What is read is validated before it is returned, so a process either starts under a
//! configuration that holds together or refuses to start and says which fields disagree. This
//! answers configuration questions only — whether the settings are consistent with each other.
//! Whether they fit the machine, such as a block pool too small for one maximum-length request,
//! is answered where the components are built and the real sizes are known.

use std::path::{Path, PathBuf};

use figment::providers::{Env, Format, Toml};
use figment::Figment;
use serde::de::DeserializeOwned;
use thiserror::Error;
use validator::{Validate, ValidationErrors};

/// The prefix every configuration environment variable carries.
const ENV_PREFIX: &str = "ATOMA_";

/// What separates nested keys in an environment variable name.
const ENV_NESTING: &str = "__";

/// Why a configuration could not be loaded.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The path names no file. Reported separately because a merged TOML source that is absent
    /// contributes nothing silently, which would surface as every field being missing.
    #[error("no configuration file at {}", .path.display())]
    Missing { path: PathBuf },
    /// The file could not be read, or what it holds does not describe this configuration.
    #[error("configuration at {} could not be read: {source}", .path.display())]
    Unreadable {
        path: PathBuf,
        #[source]
        source: Box<figment::Error>,
    },
    /// The configuration was read whole but its fields disagree.
    #[error("configuration at {} is not valid: {source}", .path.display())]
    Invalid {
        path: PathBuf,
        #[source]
        source: Box<ValidationErrors>,
    },
}

/// Reads a configuration from the TOML file at `path`, applies any `ATOMA_` environment
/// overrides, and validates the result.
///
/// # Errors
///
/// Returns [`ConfigError::Missing`] when `path` names no file, [`ConfigError::Unreadable`] when
/// what it holds does not describe `T`, and [`ConfigError::Invalid`] when the fields of `T`
/// disagree.
pub fn load<T>(path: &Path) -> Result<T, ConfigError>
where
    T: DeserializeOwned + Validate,
{
    if !path.is_file() {
        return Err(ConfigError::Missing {
            path: path.to_path_buf(),
        });
    }
    let config: T = Figment::new()
        .merge(Toml::file(path))
        .merge(Env::prefixed(ENV_PREFIX).split(ENV_NESTING))
        .extract()
        .map_err(|source| ConfigError::Unreadable {
            path: path.to_path_buf(),
            source: Box::new(source),
        })?;
    config.validate().map_err(|source| ConfigError::Invalid {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    // The closure `Jail::expect_with` takes returns `figment::Error`, which is large. The
    // signature is figment's, so the size is not ours to change.
    #![allow(clippy::result_large_err)]

    use std::path::{Path, PathBuf};

    use figment::Jail;

    use crate::config::{load, ConfigError};
    use crate::engine::EngineConfig;

    /// Every test that reads the environment runs inside a jail, which restores the variables it
    /// sets and holds a process-wide lock while it runs. A test reading configuration outside one
    /// would see whatever a jailed test running beside it had set.
    fn in_a_jail_with_the_example(test: impl FnOnce(&mut Jail, &Path)) {
        let example = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
        let source = std::fs::read_to_string(example).expect("the example configuration is there");
        Jail::expect_with(|jail| {
            jail.create_file("config.toml", &source)?;
            test(jail, Path::new("config.toml"));
            Ok(())
        });
    }

    #[test]
    fn the_example_configuration_loads() {
        in_a_jail_with_the_example(|_, path| {
            let config: EngineConfig = load(path).expect("the example configuration loads");
            assert_eq!(config.scheduler.block_size.get(), 16);
            assert_eq!(config.idle_deadline.as_millis(), 5);
        });
    }

    #[test]
    fn a_missing_file_is_named_rather_than_read_as_empty() {
        let path = PathBuf::from("does-not-exist.toml");
        let error = load::<EngineConfig>(&path).expect_err("a missing file cannot load");
        assert!(
            matches!(error, ConfigError::Missing { .. }),
            "expected a missing-file error, got {error}"
        );
    }

    #[test]
    fn an_environment_variable_overrides_the_file_it_nests_under() {
        in_a_jail_with_the_example(|jail, path| {
            jail.set_env("ATOMA_SCHEDULER__TOKEN_BUDGET", "4096");

            let config: EngineConfig = load(path).expect("the overridden configuration loads");
            assert_eq!(config.scheduler.token_budget.get(), 4096);
        });
    }

    #[test]
    fn a_batch_larger_than_the_slab_is_refused() {
        in_a_jail_with_the_example(|jail, path| {
            // The slab is set below what one step may batch, so a full batch could never be drawn
            // from it.
            jail.set_env("ATOMA_SCHEDULER__MAX_REQUESTS", "4");

            let error = load::<EngineConfig>(path).expect_err("a batch over the slab cannot load");
            assert!(
                matches!(error, ConfigError::Invalid { .. }),
                "expected a validation error, got {error}"
            );
            assert!(
                error.to_string().contains("max_batch"),
                "the error names the field: {error}"
            );
        });
    }
}

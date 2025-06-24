use home::home_dir;
use serde::Deserialize;
use crate::error::{Error, ResultWrapErr};

#[derive(Deserialize)]
pub(crate) struct Config {
    pub(crate) server: ServerConfig,
    pub(crate) lancedb: LanceDbConfig,
}

#[derive(Deserialize)]
pub(crate) struct ServerConfig {
    pub(crate) port: u16,
}

#[derive(Deserialize)]
pub(crate) struct LanceDbConfig {
    pub(crate) db_file: String
}

pub(crate) fn get_config() -> Result<Config, Error> {
    let config_path =
        home_dir()
            .ok_or_else(|| Error::from("Could not determine home directory"))?
            .join(".config").join("faris").join("config.toml");
    let config_str = std::fs::read_to_string(&config_path)
        .wrap_err(format!("Could not read config file {}", config_path.display()))?;
    toml::from_str(&config_str)
        .wrap_err(format!("Could not parse config file {}", config_path.display()))
}

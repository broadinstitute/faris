use axum::Router;
use axum::routing::get;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use crate::error::Error;
use crate::error::ResultWrapErr;

mod error;
mod config;

pub fn run() -> Result<(), Error> {
    let config = config::get_config().wrap_err("Error loading configuration")?;
    let endpoint = format!("0.0.0.0:{}", config.server.port);
    let router = Router::new().route("/ping", get(ping));
    let runtime = Runtime::new().wrap_err("Error initializing Tokio runtime")?;
    runtime.block_on(async {
        let listener =
            TcpListener::bind(endpoint)
                .await
                .wrap_err("Error binding to port")?;
        axum::serve(listener, router)
            .await
            .wrap_err("Error setting up web service")?;
        Ok::<(), Error>(())
    })?;
    Ok(())
}

async fn ping() -> &'static str {
    "Hello, world!"
}


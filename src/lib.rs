use crate::config::ModelConfig;
use crate::error::Error;
use crate::error::ResultWrapErr;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert;
use candle_transformers::models::bert::BertModel;
use chrono::Local;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, TermLogger, TerminalMode};
use std::fs;
use std::fs::File;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use crate::embed::{calculate_embedding, get_bert_model, get_device, get_tokenizer};

mod config;
mod error;
mod embed;

#[derive(Clone)]
pub(crate) struct AppState {
    tokenizer: Arc<Tokenizer>,
    device: Arc<Device>,
    bert_model: Arc<BertModel>,
}

pub fn run() -> Result<(), Error> {
    TermLogger::init(
        LevelFilter::Info,
        simplelog::Config::default(),
        TerminalMode::Stdout,
        ColorChoice::Auto,
    )
    .wrap_err("Error initializing logger")?;
    let config = config::get_config().wrap_err("Error loading configuration")?;
    let endpoint = format!("0.0.0.0:{}", config.server.port);
    let app_state = init_app_state(&config.model)?;
    let router = Router::new()
        .route("/ping", get(ping))
        .route("/embedding/{term}", get(get_embedding))
        .with_state(app_state);
    let runtime = Runtime::new().wrap_err("Error initializing Tokio runtime")?;
    runtime.block_on(async {
        let listener = TcpListener::bind(endpoint)
            .await
            .wrap_err("Error binding to port")?;
        info!("Server listening on {}", listener.local_addr().unwrap());
        axum::serve(listener, router)
            .await
            .wrap_err("Error setting up web service")?;
        Ok::<(), Error>(())
    })?;
    Ok(())
}

async fn ping() -> String {
    info!("Received ping request");
    format!(
        "Faris server is running as of {}",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    )
}
async fn get_embedding(
    State(app_state): State<AppState>, Path(term): Path<String>,
) -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    info!("Received request for embedding of term: {term}");
    match calculate_embedding(&app_state, &term) {
        Ok(embedding) => Ok(Json(embedding)),
        Err(e) => {
            info!("Error calculating embedding for term {term}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}


fn init_app_state(config: &ModelConfig) -> Result<AppState, Error> {
    let tokenizer = Arc::new(get_tokenizer(config)?);
    let device = Arc::new(get_device()?);
    let bert_model = Arc::new(get_bert_model(config, &device)?);
    Ok(AppState { tokenizer, device, bert_model, })
}


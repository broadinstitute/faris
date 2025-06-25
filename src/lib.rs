use std::fs;
use std::fs::File;
use std::sync::Arc;
use axum::extract::{Path, State};
use crate::error::Error;
use crate::error::ResultWrapErr;
use axum::{Json, Router};
use axum::http::StatusCode;
use axum::routing::get;
use candle_core::{DType, Device, Tensor};
use chrono::Local;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, TermLogger, TerminalMode};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use crate::config::ModelConfig;
use candle_transformers::models::bert;
use candle_transformers::models::bert::BertModel;
use candle_nn::VarBuilder;

mod config;
mod error;

#[derive(Clone)]
pub(crate) struct AppState {
    tokenizer: Arc<Tokenizer>,
    device: Arc<Device>,
    bert_model: Arc<BertModel>
}

pub fn run() -> Result<(), Error> {
    TermLogger::init(
        LevelFilter::Trace,
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
        "Pong! Current time: {}",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    )
}
async fn get_embedding(State(app_state): State<AppState>, Path(term): Path<String>)
    -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    info!("Received request for embedding of term: {}", term);
    let encoding = app_state.tokenizer
        .encode(term.clone(), true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let ids = encoding.get_ids().to_vec();
    let input = Tensor::new(&ids[..], &app_state.device)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .unsqueeze(0).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let seq_len = input.shape().dims()[1];
    let token_type_ids = Tensor::zeros(&[1, seq_len], input.dtype(), input.device())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let output = app_state.bert_model.forward(&input, &token_type_ids, None)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let cls_embedding =
        output.get(0).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .get(0).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .to_vec1::<f32>().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json::from(cls_embedding))
}

fn init_app_state(config: &ModelConfig) -> Result<AppState, Error> {
    let tokenizer =
        Arc::new(Tokenizer::from_file(&config.tokenizer_file)
            .map_err(|e| Error::rewrap(format!("Error loading tokenizer from file {}",
                                               config.tokenizer_file), e))?);
    let bert_config: bert::Config =
        serde_json::from_reader(File::open(&config.config_file)
            .wrap_err(format!("Error opening {}", config.config_file))?)
            .wrap_err(format!("Error parsing {}", config.config_file))?;
    let device = Arc::new(Device::Cpu);
    let dtype = DType::F32;
    let weights =
        fs::read(&config.weights_file)
            .wrap_err(format!("Error reading weights from {}", config.weights_file))?;
    let var_builder = VarBuilder::from_buffered_safetensors(weights, dtype, &device)?;
    let bert_model = Arc::new(BertModel::load(var_builder, &bert_config)?);
    Ok(AppState { tokenizer, device, bert_model })
}

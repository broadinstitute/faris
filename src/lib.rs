use crate::config::Config;
use crate::error::Error;
use crate::error::ResultWrapErr;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use candle_core::{Device};
use candle_transformers::models::bert::BertModel;
use chrono::Local;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, TermLogger, TerminalMode};
use std::sync::Arc;
use lancedb::Connection;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use crate::embed::{calculate_embedding, get_bert_model, get_device, get_tokenizer, BertModelWrap};
use crate::upload::UploadStats;

mod config;
mod error;
mod embed;
mod lance;
mod upload;

#[derive(Clone)]
pub(crate) struct AppState {
    tokenizer: Arc<Tokenizer>,
    device: Arc<Device>,
    bert_model: Arc<BertModel>,
    lance_connection: Connection,
    table_name: String,
    upload_dir: String,
    upload_stats: Arc<RwLock<UploadStats>>
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
    let runtime = Runtime::new().wrap_err("Error initializing Tokio runtime")?;
    runtime.block_on(async {
        let app_state = init_app_state(&config).await?;
        let router = Router::new()
            .route("/ping", get(ping))
            .route("/embedding/{term}", get(get_embedding))
            .route("/add/{term}", get(add_term))
            .route("/nearest/{term}", get(find_nearest))
            .route("/upload/{file_name}", get(upload_file))
            .route("/upload_stats", get(upload_stats))
            .with_state(app_state);
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

async fn add_term(
    State(app_state): State<AppState>, Path(term): Path<String>,
) -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    info!("Received request to add: {term}");
    match lance::add_if_not_exists(&app_state, &term).await {
        Ok(embedding) => {
            info!("Successfully added term {term} with embedding: [{}, {}, {} ...]",
                embedding[0], embedding[1], embedding[2]);
            Ok(Json(embedding))
        }
        Err(e) => {
            info!("Error adding term {term}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn find_nearest(
    State(app_state): State<AppState>, Path(term): Path<String>,
) -> Result<Json<Vec<lance::NearTerm>>, (StatusCode, String)> {
    info!("Received request to find nearest terms to: {term}");
    match lance::find_nearest_to(&app_state, &term, 10).await {
        Ok(terms) => Ok(Json(terms)),
        Err(e) => {
            info!("Error finding nearest terms to {term}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn upload_file(
    State(app_state): State<AppState>, Path(file_name): Path<String>,
) -> Result<String, (StatusCode, String)> { 
    info!("Received file upload request for: {file_name}");
    let stats = app_state.upload_stats.clone();
    match upload::upload_file(&app_state, file_name.clone(), stats).await {
        Ok(response) => {
            info!("Started uploading file {file_name} uploaded successfully");
            Ok(response)
        }
        Err(e) => {
            info!("Error uploading file {file_name}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn upload_stats(
    State(app_state): State<AppState>,
) -> String {
    info!("Received request for upload stats");
    app_state.upload_stats.read().await.to_string()
}

async fn init_app_state(config: &Config) -> Result<AppState, Error> {
    let model_config = &config.model;
    let tokenizer = Arc::new(get_tokenizer(model_config)?);
    let device = Arc::new(get_device()?);
    let bert_model_wrap = get_bert_model(model_config, &device)?;
    let BertModelWrap { bert_model, hidden_size } = bert_model_wrap;
    let bert_model = Arc::new(bert_model);
    let lance_connection = lance::get_connection(&config.lancedb, hidden_size).await?;
    let table_name = config.lancedb.table_name.clone();
    let upload_dir = config.server.upload_dir.clone();
    let upload_stats = Arc::new(RwLock::new(UploadStats::new()));
    Ok(AppState {
        tokenizer, device, bert_model, lance_connection, table_name, upload_dir, upload_stats
    })
}


use crate::config::Config;
use crate::error::Error;
use crate::error::ResultWrapErr;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use candle_core::{Device};
use candle_transformers::models::bert::BertModel;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, TermLogger, TerminalMode};
use std::sync::Arc;
use lancedb::Connection;
use time::OffsetDateTime;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio::sync::RwLock;
use tower_http::cors;
use tower_http::cors::CorsLayer;
use crate::embed::{calculate_embedding, get_bert_model, get_device, get_tokenizer, BertModelWrap};
use crate::lance::{NearTerm, TableStats};
use crate::upload::UploadStats;
use crate::util::format_date_time;

mod config;
mod error;
mod embed;
mod lance;
mod upload;
mod util;

#[derive(Clone)]
pub(crate) struct AppState {
    tokenizer: Arc<Tokenizer>,
    device: Arc<Device>,
    bert_model: Arc<BertModel>,
    lance_connection: Connection,
    hidden_size: usize,
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
        let cors =
            CorsLayer::new().allow_origin(cors::Any).allow_methods(cors::Any)
                .allow_headers(cors::Any);
        let table_router = Router::new()
            .route("/create_table", get(create_table))
            .route("/drop_table", get(drop_table))
            .route("/upload/{file_name}", get(upload_file_to_table))
            .route("/nearest/{term}", get(find_nearest_in_table));
        let default_router = Router::new()
            .route("/ping", get(ping))
            .route("/list_tables", get(list_tables))
            .route("/embedding/{term}", get(get_calculated_embedding))
            .route("/embedding_stored/{term}", get(get_stored_embedding))
            .route("/nearest/{term}", get(find_nearest))
            .route("/upload/{file_name}", get(upload_file))
            .route("/upload_stats", get(upload_stats));
        let router = Router::new()
            .nest("/{table}", table_router)
            .merge(default_router)
            .layer(cors)
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

async fn create_table(
    State(app_state): State<AppState>, Path(table_name): Path<String>,
) -> Result<String, (StatusCode, String)> {
    info!("Received request to create table: {table_name}");
    match lance::create_table(&app_state.lance_connection, &table_name, app_state.hidden_size)
        .await {
        Ok(_) => {
            info!("Table {table_name} created successfully");
            Ok(format!("Table {table_name} created successfully"))
        }
        Err(e) => {
            info!("Error creating table {table_name}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn drop_table(
    State(app_state): State<AppState>, Path(table_name): Path<String>,
) -> Result<String, (StatusCode, String)> {
    info!("Received request to drop table: {table_name}");
    match lance::drop_table(&app_state.lance_connection, &table_name).await {
        Ok(_) => {
            info!("Table {table_name} dropped successfully");
            Ok(format!("Table {table_name} dropped successfully"))
        }
        Err(e) => {
            info!("Error dropping table {table_name}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn list_tables(
    State(app_state): State<AppState>,
) -> Result<Json<Vec<TableStats>>, (StatusCode, String)> {
    info!("Received request to list tables");
    match lance::list_tables(&app_state.lance_connection).await {
        Ok(tables) => {
            info!("Tables listed successfully");
            Ok(Json(tables))
        }
        Err(e) => {
            info!("Error listing tables: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

async fn ping() -> String {
    info!("Received ping request");
    let now =
        OffsetDateTime::now_local().map(|dt| format_date_time(&dt))
            .unwrap_or("now".to_string());
    format!("Faris server is running as of {now}")
}
async fn get_calculated_embedding(
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

async fn get_stored_embedding(
    State(app_state): State<AppState>, Path(term): Path<String>,
) -> Result<Json<Vec<f32>>, (StatusCode, String)> {
    info!("Received request for embedding of term: {term}");
    match lance::get(&app_state, &term).await {
        Ok(embedding) => {
            match embedding {
                Some(embedding) => Ok(Json(embedding)),
                None => Err((StatusCode::NOT_FOUND,
                             format!("No embedding found for term: {term}")))
            }
        },
        Err(e) => {
            info!("Error calculating embedding for term {term}: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}


async fn find_nearest(
    State(app_state): State<AppState>, Path(term): Path<String>,
) -> Result<Json<Vec<NearTerm>>, (StatusCode, String)> {
    info!("Received request to find nearest terms to: {term}");
    let table_name = app_state.table_name.clone();
    do_find_nearest(&app_state, table_name, term).await
}

async fn find_nearest_in_table(
    State(app_state): State<AppState>, Path((table_name, term)): Path<(String, String)>,
) -> Result<Json<Vec<NearTerm>>, (StatusCode, String)> {
    info!("Received request to find nearest terms to '{term}' in table '{table_name}'");
    do_find_nearest(&app_state, table_name, term).await
}

async fn do_find_nearest(app_state: &AppState, table_name: String, term: String)
    -> Result<Json<Vec<NearTerm>>, (StatusCode, String)> {
    match lance::find_nearest_to(app_state, &table_name, &term, 1000).await {
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
    let table_name = app_state.table_name.clone();
    do_upload_file(&app_state, table_name, file_name).await
}

async fn upload_file_to_table(
    State(app_state): State<AppState>, Path((table_name, file_name)): Path<(String, String)>,
) -> Result<String, (StatusCode, String)> {
    do_upload_file(&app_state, table_name, file_name).await
}

async fn do_upload_file(app_state: &AppState, table_name: String, file_name: String)
    -> Result<String, (StatusCode, String)> {
    info!("Received request to upload file {file_name} to table {table_name}.");
    let stats = app_state.upload_stats.clone();
    match upload::upload_file(app_state, &app_state.table_name, file_name.clone(), stats).await {
        Ok(response) => {
            info!("Started uploading file {file_name} to table {table_name}.");
            Ok(response)
        }
        Err(e) => {
            info!("Error uploading file {file_name} to table {table_name}: {e}");
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
        tokenizer, device, bert_model, lance_connection, hidden_size, table_name, upload_dir,
        upload_stats
    })
}


use std::fmt::Display;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use lancedb::table::OptimizeAction;
use log::{error, info};
use serde::Deserialize;
use time::OffsetDateTime;
use tokio::sync::RwLock;
use crate::error::{Error, ResultWrapErr};
use crate::{lance, AppState};
use crate::util::format_date_time;

pub(crate) struct UploadTaskStats {
    pub(crate) file_name: String,
    pub(crate) n_terms_uploaded: usize,
    pub(crate) upload_finished: bool,
    pub(crate) indexing_finished: bool,
    pub(crate) error: Option<String>,
    pub(crate) started: OffsetDateTime,
    pub(crate) last_updated: OffsetDateTime,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct TaskStatHandle {
    i: usize,
}

pub(crate) struct UploadStats {
    tasks: Vec<UploadTaskStats>,
}

impl UploadTaskStats {
    pub(crate) fn new(file_name: String) -> Result<Self, Error> {
        let started = OffsetDateTime::now_local()
            .wrap_err("Failed to get local time")?;
        let last_updated = started;
        Ok(UploadTaskStats {
            file_name,
            n_terms_uploaded: 0,
            upload_finished: false,
            indexing_finished: false,
            error: None,
            started,
            last_updated,
        })
    }
    fn set_last_updated(&mut self) {
        if let Ok(now) = OffsetDateTime::now_local() {
            self.last_updated = now;
        } else {
            eprintln!("Failed to get local time for updating task stats");
        }
    }
    pub(crate) fn update_n_terms_uploaded(&mut self, n: usize) {
        self.n_terms_uploaded = n;
        self.set_last_updated();
    }
    pub(crate) fn mark_upload_finished(&mut self) {
        self.upload_finished = true;
        self.set_last_updated();
    }
    pub(crate) fn mark_indexing_finished(&mut self) {
        self.indexing_finished = true;
        self.set_last_updated();
    }
    pub(crate) fn mark_error(&mut self, error: String) {
        self.error = Some(error);
        self.set_last_updated();
    }
}

impl UploadStats {
    pub(crate) fn new() -> Self {
        UploadStats { tasks: Vec::new() }
    }

    pub(crate) fn add_upload(&mut self, file_name: String) -> Result<TaskStatHandle, Error> {
        let handle = TaskStatHandle { i: self.tasks.len(), };
        self.tasks.push(UploadTaskStats::new(file_name)?);
        Ok(handle)
    }
    pub(crate) fn update_task<F>(&mut self, handle: TaskStatHandle, mutator: F)
        -> Result<(), Error> where F: Fn(&mut UploadTaskStats) {
        if let Some(task) = self.tasks.get_mut(handle.i) {
            mutator(task);
            Ok(())
        } else {
            Err(Error::from("Invalid task handle"))
        }
    }
}

impl Display for UploadTaskStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Started at {}, as of {}, uploading {}", format_date_time(&self.started),
               format_date_time(&self.last_updated), self.file_name)?;
        if self.upload_finished && self.indexing_finished {
            write!(f, " is finished uploading and indexing and")?;
        } else if self.upload_finished {
            write!(f, " is finished uploading but still indexing and")?;
        } else {
            write!(f, " is in progress and so far")?;
        }
        write!(f, " {} terms have been uploaded.", self.n_terms_uploaded)?;
        if let Some(error) = &self.error {
            write!(f, " There has been an error: {error}")?;
        }
        Ok(())
    }
}

impl Display for UploadStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let now =
            OffsetDateTime::now_local()
                .map(|dt| format_date_time(&dt)).unwrap_or("now".to_string());
        writeln!(f, "As of {now} have received {} requests to upload files", self.tasks.len())?;
        for task in &self.tasks {
            writeln!(f, "{task}")?;
        }
        Ok(())
    }
}
pub(crate) async fn upload_file(app_state: &AppState, file_name: String,
                                stats: Arc<RwLock<UploadStats>>)
    -> Result<String, Error> {
    let task_handle = {
        stats.write().await.add_upload(file_name.clone())?
    };
    let message = format!("Processing request to upload {file_name}");
    let app_state = app_state.clone();
    tokio::spawn( async {
        tokio::task::spawn_blocking(move || {
            upload_spawned(app_state, file_name, stats, task_handle)
        }).await
    }.await.wrap_err("Error spawning")?);
    Ok(message)
}

async fn upload_spawned(app_state: AppState, file_name: String,
                        stats: Arc<RwLock<UploadStats>>, handle: TaskStatHandle) {
    let file_path = Path::new(&app_state.upload_dir).join(&file_name);
    let upload_result =
        try_upload(app_state, file_path, stats.clone(), handle).await;
    if let Err(error) = upload_result {
        let error = error.to_string();
        update_stats(&stats, handle, |task| {
            task.mark_error(error.clone()) }
        ).await;
    }
}

#[derive(Deserialize)]
struct UploadRecord {
    term: String,
    phenotype: Option<String>,
    gene_set: Option<String>,
    source: Option<String>,
    beta_uncorrected: Option<f32>,
}

async fn try_upload(app_state: AppState, file_path: PathBuf, stats: Arc<RwLock<UploadStats>>,
                    handle: TaskStatHandle)
    -> Result<(), Error> {
    let file = std::fs::File::open(&file_path)
        .wrap_err(format!("Failed to open file {}", file_path.display()))?;
    let mut reader =
        csv::ReaderBuilder::new().trim(csv::Trim::All).from_reader(file);
    info!("Starting to upload terms from file: {}", file_path.display());
    let mut n_terms: usize = 0;
    let mut term_buffer: Vec<String> = Vec::new();
    let mut phenotype_buffer: Vec<Option<String>> = Vec::new();
    let mut gene_set_buffer: Vec<Option<String>> = Vec::new();
    let mut source_buffer: Vec<Option<String>> = Vec::new();
    let mut beta_buffer: Vec<Option<f32>> = Vec::new();
    for record in reader.deserialize::<UploadRecord>() {
        let record: UploadRecord =
            match record {
                Ok(rec) => rec,
                Err(error) => {
                    let error = Error::wrap("Failed to read CSV record", error);
                    error!("Error reading record from file {}: {}. Skipping.",
                           file_path.display(), error);
                    continue
                },
            };
        let UploadRecord {
            term, phenotype, gene_set, source,
            beta_uncorrected
        } = record;
        term_buffer.push(term);
        phenotype_buffer.push(phenotype);
        gene_set_buffer.push(gene_set);
        source_buffer.push(source);
        beta_buffer.push(beta_uncorrected);
        const BATCH_SIZE: usize = 1000;
        if term_buffer.len() >= BATCH_SIZE {
            let n_terms_batch = term_buffer.len();
            let term_batch = mem::take(&mut term_buffer);
            let phenotype_batch = mem::take(&mut phenotype_buffer);
            let gene_set_batch = mem::take(&mut gene_set_buffer);
            let source_batch = mem::take(&mut source_buffer);
            let beta_batch = mem::take(&mut beta_buffer);
            lance::add(&app_state, term_batch, phenotype_batch, gene_set_batch, source_batch,
                       beta_batch).await?;
            n_terms += n_terms_batch;
            info!("Uploaded batch of {n_terms_batch} terms, {n_terms} total so far");
            term_buffer.clear();
            update_stats(&stats, handle, |task| {
                task.update_n_terms_uploaded(n_terms);
            }).await;
        }
    }
    if !term_buffer.is_empty() {
        let n_terms_batch = term_buffer.len();
        lance::add(&app_state, term_buffer, phenotype_buffer, gene_set_buffer, source_buffer,
                   beta_buffer).await?;
        n_terms += n_terms_batch;
        info!("Uploaded batch of {n_terms_batch} terms, {n_terms} total so far");
    }
    update_stats(&stats, handle, |task| {
        task.update_n_terms_uploaded(n_terms);
        task.mark_upload_finished();
    }).await;
    info!("Finished uploading terms from file: {}. Now indexing.", file_path.display());
    let table = app_state.lance_connection.open_table(app_state.table_name).execute().await?;
    table.optimize(OptimizeAction::All).await?;
    update_stats(&stats, handle, |task| {
        task.mark_indexing_finished();
    }).await;
    Ok(())
}

async fn update_stats<F>(stats: &Arc<RwLock<UploadStats>>, handle: TaskStatHandle, mutator: F)
where F: Fn(&mut UploadTaskStats) {
    stats.write().await.update_task(handle, mutator)
        .unwrap_or_else(|error| {
            eprintln!("Failed to update upload stats: {error}");
        });
}
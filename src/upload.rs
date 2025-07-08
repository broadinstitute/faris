use std::fmt::Display;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use time::OffsetDateTime;
use tokio::sync::RwLock;
use crate::error::{Error, ResultWrapErr};
use crate::{lance, AppState};

pub(crate) struct UploadTaskStats {
    pub(crate) file_name: String,
    pub(crate) n_terms_uploaded: usize,
    pub(crate) is_finished: bool,
    pub(crate) error: Option<String>,
    pub(crate) timestamp: OffsetDateTime
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
        let timestamp = OffsetDateTime::now_local()
            .wrap_err("Failed to get local time")?;
        Ok(UploadTaskStats {
            file_name,
            n_terms_uploaded: 0,
            is_finished: false,
            error: None,
            timestamp,
        })
    }
    pub(crate) fn update_n_terms_uploaded(&mut self, n: usize) {
        self.n_terms_uploaded = n;
    }
    pub(crate) fn mark_finished(&mut self) {
        self.is_finished = true;
    }
    pub(crate) fn mark_error(&mut self, error: String) {
        self.error = Some(error);
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
    pub(crate) fn update_task<F>(&mut self, handle: &TaskStatHandle, mutator: F)
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
        write!(f, "As of {}, uploading {}", self.timestamp, self.file_name)?;
        if self.is_finished {
            write!(f, " is finished and")?;
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
                .map(|dt| dt.to_string()).unwrap_or("now".to_string());
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
    tokio::spawn(async move {
        upload_spawned(app_state, file_name, stats, task_handle).await
    });
    Ok(message)
}

async fn upload_spawned(app_state: AppState, file_name: String,
                        stats: Arc<RwLock<UploadStats>>, handle: TaskStatHandle) {
    let file_path = Path::new(&app_state.upload_dir).join(&file_name);
    let upload_result = try_upload(app_state, file_path, stats.clone(), handle).await;
    if let Err(error) = upload_result {
        let error = error.to_string();
        stats.write().await.update_task(&handle, |task| {
            task.mark_error(error.clone());
        }).unwrap_or_else(|e| {
            eprintln!("Failed to update upload stats: {}", e);
        });
    }
}

async fn try_upload(app_state: AppState, file_path: PathBuf, stats: Arc<RwLock<UploadStats>>,
                    handle: TaskStatHandle)
    -> Result<(), Error> {
    let file = std::fs::File::open(&file_path)
        .wrap_err(format!("Failed to open file {}", file_path.display()))?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));
    for (i, record) in reader.records().enumerate() {
        let record = record.wrap_err("Failed to read CSV record")?;
        if let Some(term) = record.get(0) {
            lance::add_if_not_exists(&app_state, term).await?;
        }
    }
    Ok(())
}
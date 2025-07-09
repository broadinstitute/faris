use crate::config::LanceDbConfig;
use crate::error::{Error, ResultWrapErr};
use crate::{embed, AppState};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use futures::StreamExt;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection};
use serde::Serialize;
use std::sync::Arc;
use lancedb::index::Index;
use log::{info, warn};

pub(crate) const TERM_COLUMN: &str = "term";
pub(crate) const EMBEDDING_COLUMN: &str = "embedding";

pub(crate) async fn get_connection(
    config: &LanceDbConfig, hidden_size: usize,
) -> Result<Connection, Error> {
    let db_file = &config.db_file;
    let connection = connect(db_file)
        .execute()
        .await
        .wrap_err(format!("Failed to connect to LanceDB at {db_file}"))?;
    create_table_if_not_exists(&connection, &config.table_name, hidden_size)
        .await
        .wrap_err(format!(
            "Failed to ensure table exists in LanceDB at {db_file}"
        ))?;
    Ok(connection)
}

async fn create_table_if_not_exists(
    connection: &Connection, table_name: &str, hidden_size: usize,
) -> Result<(), Error> {
    let table_names = connection
        .table_names()
        .execute()
        .await
        .wrap_err("Failed to list tables.".to_string())?;
    if !table_names.iter().any(|name| name == table_name) {
        create_table(connection, table_name, hidden_size).await?
    }
    let table = connection
        .open_table(table_name)
        .execute()
        .await
        .wrap_err(format!("Failed to open table {table_name}"))?;
    try_creating_index(&table).await?;
    Ok(())
}

async fn create_table(
    connection: &Connection, table_name: &str, hidden_size: usize,
) -> Result<(), Error> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(TERM_COLUMN, DataType::Utf8, false),
        Field::new(
            "EMBEDDING_COLUMN",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                hidden_size as i32,
            ),
            false,
        ),
    ]));
    let table =
        connection
            .create_empty_table(table_name, schema)
            .execute()
            .await
            .wrap_err(format!("Failed to create table {table_name}"))?;
    try_creating_index(&table).await?;
    Ok(())
}

pub(crate) async fn try_creating_index(table: &lancedb::table::Table) -> Result<(), Error> {
    match table.create_index(&[EMBEDDING_COLUMN], Index::Auto).execute().await {
        Ok(_) => {
            info!("Index created successfully.");
        }
        Err(e) => {
            warn!("Failed to create index: {e}");
        }
    }
    Ok(())
}

async fn add(app_state: &AppState, term: &str) -> Result<Vec<f32>, Error> {
    let embedding = crate::embed::calculate_embedding(app_state, term)
        .wrap_err(format!("Failed to calculate embedding for term '{term}'"))?;
    let dim = embedding.len();
    let terms = Arc::new(StringArray::from(vec![term])) as ArrayRef;
    let values = Arc::new(Float32Array::from(embedding.clone())) as ArrayRef;
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let embeddings = Arc::new(FixedSizeListArray::try_new(field, dim as i32, values, None)
        .expect("Failed to build FixedSizeListArray")) as ArrayRef;
    let batch = RecordBatch::try_from_iter(vec![
        ("term", terms),
        ("embedding", embeddings),
    ])?;
    let table = app_state.lance_connection
        .open_table(&app_state.table_name)
        .execute()
        .await
        .wrap_err(format!("Could not open table {}", app_state.table_name))?;
    let schema = batch.schema();
    let iter = vec![Ok(batch)].into_iter();
    let batch_reader = RecordBatchIterator::new(iter, schema);
    table.add(batch_reader)
        .execute()
        .await
        .wrap_err(format!("Failed to add record to table {}", app_state.table_name))?;
    Ok(embedding)
}

async fn get(
    app_state: &AppState, term: &str,
) -> Result<Option<Vec<f32>>, Error> {
    let table = &app_state.lance_connection
        .open_table(&app_state.table_name)
        .execute()
        .await
        .wrap_err(format!("Could not open table {}", app_state.table_name))?;
    let escaped_term = term.replace("'", "''");
    let mut results = table
        .query()
        .only_if(format!("term = '{escaped_term}'"))
        .execute()
        .await
        .wrap_err(format!("Could not query table {}", app_state.table_name))?;
    if let Some(record_batch) = results.next().await {
        let record_batch =
            record_batch.wrap_err(format!("Failed to retrieve record batch for term '{term}'"))?;
        let embedding_column = record_batch
            .column_by_name("embedding")
            .ok_or_else(|| {
                Error::from(format!("No embedding column in results for term '{term}'"))
            })?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                Error::from(format!(
                    "Embedding column is not a FixedSizeListArray for term '{term}'"
                ))
            })?
            .iter()
            .next()
            .unwrap()
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                Error::from(format!("Embedding is not a Float32Array for term '{term}'"))
            })?.iter().map(|x| x.unwrap_or(f32::NAN)).collect::<Vec<f32>>();
        Ok(Some(embedding_column))
    } else {
        Ok(None)
    }
}

pub(crate) struct MaybeAdded {
    pub(crate) embedding: Vec<f32>,
    pub(crate) was_added: bool,
}

pub(crate) async fn add_if_not_exists(app_state: &AppState, term: &str)
    -> Result<MaybeAdded, Error> {
    if let Some(embedding) = get(app_state, term).await? {
        Ok(MaybeAdded { embedding, was_added: false } )
    } else {
        add(app_state, term).await.map(|embedding| {
            MaybeAdded { embedding, was_added: true }
        })
    }
}

#[derive(Serialize)]
pub(crate) struct NearTerm {
    pub term: String,
    pub distance: f32,
}

pub(crate) async fn find_nearest_to(app_state: &AppState, term: &str, k: usize)
    -> Result<Vec<NearTerm>, Error> {
    let table = &app_state.lance_connection
        .open_table(&app_state.table_name)
        .execute()
        .await
        .wrap_err(format!("Could not open table {}", app_state.table_name))?;
    let embedding = embed::calculate_embedding(app_state, term)
        .wrap_err(format!("Failed to calculate embedding for term '{term}'"))?;
    let mut nearest_batch_stream =
        table.query().nearest_to(embedding)?.limit(k).execute()
        .await
        .wrap_err(format!("Failed to find nearest neighbors for term '{term}'"))?;
    let mut nearest_terms: Vec<NearTerm> = Vec::new();
    while let Some(batch) = nearest_batch_stream.next().await {
        let batch =
            batch.wrap_err(
                format!("Failed to retrieve batch for nearest neighbors of term '{term}'")
            )?;
        let terms_column = batch
            .column_by_name("term")
            .ok_or_else(|| Error::from(
                format!("No term column in results for term '{term}'"))
            )?;
        let terms_array = terms_column.as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| Error::from(
                format!("Term column is not a StringArray for term '{term}'"))
            )?;
        let distances_column = batch
            .column_by_name("__distance__")
            .ok_or_else(|| Error::from(
                format!("No distance column in results for term '{term}'"))
            )?;
        let distances_array =
            distances_column.as_any().downcast_ref::<Float32Array>().ok_or_else(||
                Error::from(
                    format!("Distance column is not a Float32Array for term '{term}'")
                )
            )?;
        terms_array.iter().zip(distances_array.iter())
            .for_each(|(opt, dist)| {
                let term = opt.map(|s| s.to_string()).unwrap_or_default();
                let distance = dist.unwrap_or(f32::NAN);
                nearest_terms.push(NearTerm { term, distance });
            });
    }
    nearest_terms.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    Ok(nearest_terms)
}

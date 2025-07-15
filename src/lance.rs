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
pub(crate) const PHENOTYPE_COLUMN: &str = "phenotype";
pub(crate) const GENE_SET_COLUMN: &str = "gene_set";
pub(crate) const EMBEDDING_COLUMN: &str = "embedding";
pub(crate) const DISTANCE_COLUMN: &str = "_distance";

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
    let schema = table.schema().await?;
    info!("{schema:#?}");
    let mut stream = table.query().limit(5).execute().await?;
    while let Some(batch_result) = stream.next().await {
        let batch: RecordBatch = batch_result?;
        println!("{batch:?}");
    }    try_creating_index(&table).await?;
    Ok(())
}

async fn create_table(
    connection: &Connection, table_name: &str, hidden_size: usize,
) -> Result<(), Error> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(TERM_COLUMN, DataType::Utf8, false),
        Field::new(PHENOTYPE_COLUMN, DataType::Utf8, true),
        Field::new(GENE_SET_COLUMN, DataType::Utf8, true),
        Field::new(
            EMBEDDING_COLUMN,
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

pub(crate) async fn add(app_state: &AppState, terms: Vec<String>, phenotypes: Vec<Option<String>>,
                        gene_sets: Vec<Option<String>>) -> Result<Vec<Vec<f32>>, Error> {
    if terms.is_empty() {
        Ok(Vec::new())
    } else {
        let embeddings: Vec<Vec<f32>> = terms.iter()
            .map(|t| {
                embed::calculate_embedding(app_state, t)
                    .wrap_err(format!("Failed to calculate embedding for term '{t}'"))
            }).collect::<Result<Vec<Vec<f32>>, Error>>()?;
        let dim = embeddings[0].len() as i32;
        let embeddings_flat: Vec<f32> = embeddings.iter()
            .flat_map(|e| e.iter().cloned())
            .collect();
        let terms_ref = Arc::new(StringArray::from(terms)) as ArrayRef;
        let phenotypes_ref = Arc::new(StringArray::from(phenotypes)) as ArrayRef;
        let gene_sets_ref = Arc::new(StringArray::from(gene_sets)) as ArrayRef;
        let values = Arc::new(Float32Array::from(embeddings_flat)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let embeddings_flat_ref =
            Arc::new(FixedSizeListArray::try_new(field, dim, values, None)
            .expect("Failed to build FixedSizeListArray")) as ArrayRef;
        let batch = RecordBatch::try_from_iter(vec![
            ("term", terms_ref),
            ("phenotype", phenotypes_ref),
            ("gene_set", gene_sets_ref),
            ("embedding", embeddings_flat_ref),
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
        Ok(embeddings)
    }
}

pub(crate) async fn get(
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

#[derive(Serialize)]
pub(crate) struct NearTerm {
    pub term: String,
    pub phenotype: Option<String>,
    pub gene_set: Option<String>,
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
        let terms_array = get_string_column(term, &batch, TERM_COLUMN)?;
        let phenotypes_array = get_string_column(term, &batch, PHENOTYPE_COLUMN)?;
        let gene_sets_array = get_string_column(term, &batch, GENE_SET_COLUMN)?;
        let distances_array = get_float_array_column(term, &batch, DISTANCE_COLUMN)?;
        for i in 0..terms_array.len() {
            let term = get_string_value(term, terms_array, i)?;
            let phenotype = get_opt_string_value(phenotypes_array, i);
            let gene_set = get_opt_string_value(gene_sets_array, i);
            let distance = get_f32_value(distances_array, i);
            nearest_terms.push(NearTerm { term, phenotype, gene_set, distance });
        }
    }
    nearest_terms.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    Ok(nearest_terms)
}

fn get_f32_value(distances_array: &Float32Array, i: usize) -> f32 {
    if distances_array.is_null(i) {
        f32::NAN
    } else {
        distances_array.value(i)
    }
}

fn get_opt_string_value(phenotypes_array: &StringArray, i: usize) -> Option<String> {
    if phenotypes_array.is_null(i) {
        None
    } else {
        Some(phenotypes_array.value(i).to_string())
    }
}

fn get_string_value(term: &str, terms_array: &StringArray, i: usize) -> Result<String, Error> {
    if terms_array.is_null(i) {
        Err(Error::from(
            format!("Null term found in results for nearest neighbors of term '{term}'")
        ))
    } else {
        Ok(terms_array.value(i).to_string())
    }
}

fn get_string_column<'a>(term: &str, batch: &'a RecordBatch, column_name: &str)
                         -> Result<&'a StringArray, Error> {
    let terms_column = batch
        .column_by_name(column_name)
        .ok_or_else(|| Error::from(
            format!("No '{column_name}' column in results for term '{term}'"))
        )?;
    let terms_array = terms_column.as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| Error::from(
            format!("Column '{column_name}' is not a StringArray for term '{term}'"))
        )?;
    Ok(terms_array)
}

fn get_float_array_column<'a>(term: &str, batch: &'a RecordBatch, column_name: &str)
    -> Result<&'a Float32Array, Error> {
    let distances_column = batch
        .column_by_name(column_name)
        .ok_or_else(|| Error::from(
            format!("No '{column_name}' column in results for term '{term}'"))
        )?;
    let distances_array =
        distances_column.as_any().downcast_ref::<Float32Array>().ok_or_else(||
            Error::from(
                format!("Column '{column_name}' is not a Float32Array for term '{term}'")
            )
        )?;
    Ok(distances_array)
}


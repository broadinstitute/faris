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
use log::info;

pub(crate) const TERM_COLUMN: &str = "term";
pub(crate) const PHENOTYPE_COLUMN: &str = "phenotype";
pub(crate) const GENE_SET_COLUMN: &str = "gene_set";
pub(crate) const SOURCE_COLUMN: &str = "source";
pub(crate) const BETA_UNCORRECTED_COLUMN: &str = "beta_uncorrected";
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
    }
    Ok(())
}

pub(crate) async fn create_table(
    connection: &Connection, table_name: &str, hidden_size: usize,
) -> Result<(), Error> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(TERM_COLUMN, DataType::Utf8, false),
        Field::new(PHENOTYPE_COLUMN, DataType::Utf8, true),
        Field::new(GENE_SET_COLUMN, DataType::Utf8, true),
        Field::new(SOURCE_COLUMN, DataType::Utf8, true),
        Field::new(BETA_UNCORRECTED_COLUMN, DataType::Float32, true),
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
    if let Err(error) = try_creating_index(&table).await {
        info!("Failed to create index on table {table_name}: {error}");
    }
    Ok(())
}

#[derive(Serialize)]
pub(crate) struct TableStats {
    name: String,
    n_rows: usize,
}

pub(crate) async fn list_tables(connection: &Connection) -> Result<Vec<TableStats>, Error> {
    let table_names = connection.table_names().execute()
        .await
        .wrap_err("Failed to list tables.".to_string())?;
    let table_stats =
        table_names.into_iter()
        .map(async |name| {
            let n_rows = connection
                .open_table(&name)
                .execute()
                .await
                .wrap_err(format!("Failed to open table {name}"))?
                .count_rows(None)
                .await?;
            Ok(TableStats { name, n_rows })
        });
    futures::future::try_join_all(table_stats).await
}

pub(crate) async fn drop_table(connection: &Connection, table_name: &str) -> Result<(), Error> {
        connection.drop_table(table_name).await
            .wrap_err(format!("Failed to drop table {table_name}"))?;
    Ok(())
}

pub(crate) async fn try_creating_index(table: &lancedb::table::Table) -> Result<(), Error> {
    let indices = table.list_indices().await?;
    let index_exists =
        indices.iter().any(|idx| idx.columns.contains(&EMBEDDING_COLUMN.to_string()));
    if !index_exists {
        table.create_index(&[EMBEDDING_COLUMN], Index::Auto)
            .execute().await?;
    }
    Ok(())
}

pub(crate) async fn add(app_state: &AppState, table_name: &str, terms: Vec<String>,
                        phenotypes: Vec<Option<String>>, gene_sets: Vec<Option<String>>,
                        sources: Vec<Option<String>>, beta_uncorrecteds: Vec<Option<f32>>)
    -> Result<Vec<Vec<f32>>, Error> {
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
        let sources_ref = Arc::new(StringArray::from(sources)) as ArrayRef;
        let beta_uncorrecteds_ref =
            Arc::new(Float32Array::from(beta_uncorrecteds)) as ArrayRef;
        let values = Arc::new(Float32Array::from(embeddings_flat)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let embeddings_flat_ref =
            Arc::new(FixedSizeListArray::try_new(field, dim, values, None)
            .expect("Failed to build FixedSizeListArray")) as ArrayRef;
        let batch = RecordBatch::try_from_iter(vec![
            (TERM_COLUMN, terms_ref),
            (PHENOTYPE_COLUMN, phenotypes_ref),
            (GENE_SET_COLUMN, gene_sets_ref),
            (EMBEDDING_COLUMN, embeddings_flat_ref),
            (SOURCE_COLUMN, sources_ref),
            (BETA_UNCORRECTED_COLUMN, beta_uncorrecteds_ref),
        ])?;
        let table = app_state.lance_connection
            .open_table(table_name)
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
    pub source: Option<String>,
    pub beta_uncorrected: Option<f32>,
    pub distance: f32,
}

pub(crate) async fn find_nearest_to(app_state: &AppState, table_name: &str, term: &str, k: usize)
    -> Result<Vec<NearTerm>, Error> {
    let table = &app_state.lance_connection
        .open_table(table_name)
        .execute()
        .await
        .wrap_err(format!("Could not open table {table_name}"))?;
    let embedding = embed::calculate_embedding(app_state, term)
        .wrap_err(format!("Failed to calculate embedding for term '{term}'"))?;
    let mut nearest_batch_stream =
        table.query().nearest_to(embedding)?.limit(k).execute()
        .await
        .wrap_err(format!("Failed to find nearest neighbors for term '{term}' using table '{table_name}'"))?;
    let mut nearest_terms: Vec<NearTerm> = Vec::new();
    while let Some(batch) = nearest_batch_stream.next().await {
        let batch =
            batch.wrap_err(
                format!("Failed to retrieve batch for nearest neighbors of term '{term}' from table '{table_name}'.")
            )?;
        let terms_array = get_string_column(term, &batch, TERM_COLUMN)?;
        let phenotypes_array = get_string_column(term, &batch, PHENOTYPE_COLUMN)?;
        let gene_sets_array = get_string_column(term, &batch, GENE_SET_COLUMN)?;
        let sources_array = get_string_column(term, &batch, SOURCE_COLUMN)?;
        let beta_uncorrecteds_array =
            get_float_array_column(term, &batch, BETA_UNCORRECTED_COLUMN)?;
        let distances_array = get_float_array_column(term, &batch, DISTANCE_COLUMN)?;
        for i in 0..terms_array.len() {
            let term = get_string_value(term, terms_array, i)?;
            let phenotype = get_opt_string_value(phenotypes_array, i);
            let gene_set = get_opt_string_value(gene_sets_array, i);
            let source = get_opt_string_value(sources_array, i);
            let beta_uncorrected = get_f32_opt_value(beta_uncorrecteds_array, i);
            let distance = get_f32_value(distances_array, i);
            let near_term =
                NearTerm { term, phenotype, gene_set, distance, source, beta_uncorrected };
            nearest_terms.push(near_term);
        }
    }
    nearest_terms.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    Ok(nearest_terms)
}

fn get_f32_opt_value(beta_uncorrecteds_array: &Float32Array, i: usize) -> Option<f32> {
    if beta_uncorrecteds_array.is_null(i) {
        None
    } else {
        Some(beta_uncorrecteds_array.value(i))
    }
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


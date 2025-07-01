use crate::AppState;
use crate::config::LanceDbConfig;
use crate::error::{Error, ResultWrapErr};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use futures::StreamExt;
use lancedb::arrow::IntoArrowStream;
use lancedb::arrow::arrow_schema::{DataType, Field, FieldRef, Schema};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, connect};
use std::sync::Arc;

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
    Ok(())
}

async fn create_table(
    connection: &Connection, table_name: &str, hidden_size: usize,
) -> Result<(), Error> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("term", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                hidden_size as i32,
            ),
            false,
        ),
    ]));
    connection
        .create_empty_table(table_name, schema)
        .execute()
        .await
        .wrap_err(format!("Failed to create table {table_name}"))?;
    Ok(())
}

async fn add(
    connection: &Connection, table_name: &str, term: &str, app_state: &AppState,
) -> Result<Vec<f32>, Error> {
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
    let table = connection
        .open_table(table_name)
        .execute()
        .await
        .wrap_err(format!("Could not open table {table_name}"))?;
    let schema = batch.schema();
    let iter = vec![Ok(batch)].into_iter();
    let batch_reader = RecordBatchIterator::new(iter, schema);
    table.add(batch_reader)
        .execute()
        .await
        .wrap_err(format!("Failed to add record to table {table_name}"))?;
    Ok(embedding)
}

async fn get(
    connection: &Connection, table_name: &str, term: &str,
) -> Result<Option<Vec<f32>>, Error> {
    let table = connection
        .open_table(table_name)
        .execute()
        .await
        .wrap_err(format!("Could not open table {table_name}"))?;
    let mut results = table
        .query()
        .only_if(format!("term = '{}'", term))
        .execute()
        .await
        .wrap_err(format!("Could not query table {table_name}"))?;
    if let Some(record_batch) = results.next().await {
        let record_batch =
            record_batch.wrap_err(format!("Failed to retrieve record batch for term '{term}'"))?;
        let term = record_batch
            .column_by_name("term")
            .ok_or_else(|| Error::from(format!("No term column in results for term '{term}'")))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                Error::from(format!(
                    "Term column is not a StringArray for term '{term}'"
                ))
            })?
            .iter()
            .next()
            .unwrap()
            .unwrap()
            .to_string();
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

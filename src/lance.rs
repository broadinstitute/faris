use crate::config::LanceDbConfig;
use crate::error::{Error, ResultWrapErr};
use crate::AppState;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection};
use std::sync::Arc;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};

pub(crate) async fn get_connection(config: &LanceDbConfig, hidden_size: usize)
    -> Result<Connection, Error> {
    let db_file = &config.db_file;
    let connection =
        connect(db_file).execute().await
            .wrap_err(format!("Failed to connect to LanceDB at {db_file}"))?;
    create_table_if_not_exists(&connection, &config.table_name, hidden_size).await
        .wrap_err(format!("Failed to ensure table exists in LanceDB at {db_file}"))?;
    Ok(connection)
}

async fn create_table_if_not_exists(connection: &Connection, table_name: &str, hidden_size: usize)
                                    -> Result<(), Error> {
    let table_names = connection.table_names().execute().await.wrap_err(
        "Failed to list tables.".to_string()
    )?;
    if !table_names.iter().any(|name| name == table_name) {
        create_table(connection, table_name, hidden_size).await?
    }
    Ok(())
}

async fn create_table(connection: &Connection, table_name: &str, hidden_size: usize)
               -> Result<(), Error> {
    let schema =
        Arc::new(Schema::new(vec![
            Field::new("term", DataType::Utf8, false),
            Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                hidden_size as i32), false)
        ]));
    connection.create_empty_table(table_name, schema).execute().await
        .wrap_err(format!("Failed to create table {table_name}"))?;
    Ok(())
}

async fn add(connection: &Connection, table_name: &str, term: &str, app_state: &AppState)
    -> Result<Vec<f32>, Error> {
    let embedding = crate::embed::calculate_embedding(app_state, term)
        .wrap_err(format!("Failed to calculate embedding for term '{term}'"))?;
    let table = connection.open_table(table_name).execute().await
        .wrap_err(format!("Could not open table {table_name}"))?;
    todo!()
}

async fn get(connection: &Connection, table_name: &str, term: &str)
    -> Result<Option<Vec<f32>>, Error> {
    let table =
        connection.open_table(table_name).execute().await
            .wrap_err(format!("Could not open table {table_name}"))?;
    let results =
        table.query().only_if(format!("term = '{}'", term)).execute().await
        .wrap_err(format!("Could not query table {table_name}"))?;
    if let Some(record_batch) = results {
        let embedding_column = record_batch.column_by_name("embedding")
            .ok_or_else(||
                Error::from(format!("No embedding column in results for term '{term}'"))
            )?;
        let embedding_array = embedding_column.as_fixed_size_list()
            .ok_or_else(||
                Error::from(
                    format!("Embedding column is not a fixed size list for term '{term}'")
                )
            )?;
        let embedding = embedding_array.values().iter()
            .map(|value| value.as_float32().unwrap())
            .collect::<Vec<f32>>();
        Ok(Some(embedding))
    } else {
        Ok(None)
    }
}

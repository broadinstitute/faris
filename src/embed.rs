use candle_transformers::models::bert;
use candle_transformers::models::bert::BertModel;
use crate::AppState;
use crate::config::ModelConfig;
use crate::error::Error;
use std::fs::File;
use std::fs;
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilder;
use tokenizers::tokenizer::Tokenizer;
use crate::error::ResultWrapErr;

pub(crate) fn get_tokenizer(config: &ModelConfig) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(&config.tokenizer_file).map_err(|e| {
        Error::rewrap(
            format!("Error loading tokenizer from file {}", config.tokenizer_file), e,
        )
    })
}

pub(crate) fn get_device() -> Result<Device, Error> {
    Ok(Device::Cpu)
}

pub(crate) fn get_bert_model(config: &ModelConfig, device: &Device) -> Result<BertModel, Error> {
    let bert_config: bert::Config = serde_json::from_reader(
        File::open(&config.config_file)
            .wrap_err(format!("Error opening {}", config.config_file))?,
    )
    .wrap_err(format!("Error parsing {}", config.config_file))?;
    let dtype = DType::F32;
    let weights = fs::read(&config.weights_file).wrap_err(format!(
        "Error reading weights from {}",
        config.weights_file
    ))?;
    let var_builder = VarBuilder::from_buffered_safetensors(weights, dtype, device)?;
    let bert_model = BertModel::load(var_builder, &bert_config)?;
    Ok(bert_model)
}

pub(crate) fn calculate_embedding(app_state: &AppState, term: &str) -> Result<Vec<f32>, Error> {
    let encoding = app_state
        .tokenizer
        .encode(term, true)
        .map_err(|e| Error::rewrap(format!("Error encoding term {term}"), e))?;
    let ids = encoding.get_ids().to_vec();
    let input = Tensor::new(&ids[..], &app_state.device)?.unsqueeze(0)?;
    let seq_len = input.shape().dims()[1];
    let token_type_ids = Tensor::zeros(&[1, seq_len], input.dtype(), input.device())?;
    let output = app_state
        .bert_model
        .forward(&input, &token_type_ids, None)?;
    let cls_embedding = output.get(0)?.get(0)?.to_vec1::<f32>()?;
    Ok(cls_embedding)
}
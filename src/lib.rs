use tokio::runtime::Runtime;
use crate::error::Error;
use crate::error::ResultWrapErr;

mod error;

pub fn run() -> Result<(), Error> {
    let runtime = Runtime::new().wrap_err("Error initializing Tokio runtime")?;
    runtime.block_on(async {
        // Your async code here
    });
    Ok(())
}


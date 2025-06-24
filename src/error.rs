use std::fmt::{Debug, Display, Formatter};

pub struct Error {
    pub message: String,
    source: Option<Box<dyn std::error::Error>>,
}

impl Error {
    pub fn new<S>(message: S, source: Option<Box<dyn std::error::Error>>) -> Self
    where
        S: Into<String>,
    {
        Error {
            message: message.into(),
            source,
        }
    }

    pub fn with_source<S, E>(message: S, source: E) -> Self
    where
        S: Into<String>,
        E: std::error::Error + 'static,
    {
        Error {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
    pub fn wrap<S, E>(message: S, source: E) -> Self
    where
        S: Into<String>,
        E: std::error::Error + 'static,
    {
        Error {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl Debug for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(source) = &self.source {
            write!(f, "{}: {}", self.message, source)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref()
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Error::new(message.to_string(), None)
    }
}
impl From<String> for Error {
    fn from(message: String) -> Self {
        Error::new(message, None)
    }
}

pub trait ResultWrapErr<T, E: std::error::Error + 'static> {
    fn wrap_err<S>(self, message: S) -> Result<T, Error> where S: Into<String>;
}

impl<T, E: std::error::Error + 'static> ResultWrapErr<T, E> for Result<T, E> {
    fn wrap_err<S>(self, message: S) -> Result<T, Error>
    where S: Into<String> { self.map_err(|e| Error::wrap(message, e)) }
}
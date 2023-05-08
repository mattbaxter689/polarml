use polars::prelude::*;
use std::path::Path;

//Generic reading in csv function
pub fn read_csv(path: &str) -> PolarsResult<DataFrame> {
    CsvReader::from_path(path)?.has_header(true).finish()
}

pub fn check_model_dir() -> bool {
    if Path::new("model/").is_dir() {
        return true
    }

    return false
}

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use polars::prelude::DataFrame;
use polars::datatypes::Float64Type;
use ndarray::s;
use std::fs::File;
use std::io::{Write};

use crate::fileops::check_model_dir;

pub fn fit_linfa(frame: &DataFrame) {
    let nd_frame = frame.to_ndarray::<Float64Type>().unwrap();

    let (records, targets) = (
        nd_frame.slice(s![.., 0..13]).to_owned(),
        nd_frame.column(13).to_owned(),
    ); 

    let data = Dataset::new(records, targets);
    println!("{:?}", data);

    let model = LinearRegression::default().fit(&data).unwrap();

    if check_model_dir() {
        println!("\nSaving model");
    
        let reg_bytes = bincode::serialize(&model).expect("Issue serializing model");
        File::create("model/linfa_reg.model")
            .and_then(|mut f| f.write_all(&reg_bytes))
            .expect("Can not persist model");
    }
}

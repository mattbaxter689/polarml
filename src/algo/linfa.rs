use linfa::metrics::SingleTargetRegression;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use polars::datatypes::Float64Type;
use polars::prelude::DataFrame;
use ndarray::{s, Array, Array2, ArrayBase, OwnedRepr, Dim};
use std::fs::File;
use std::io::Write;

use crate::fileops::check_model_dir;

fn linfa_split(arr: Array2<f64>) -> Result<DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, 
   ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>>, Error >{


    let (records, targets) = (
        arr.slice(s![.., 0..13]).to_owned(),
        arr.column(13).to_owned(),
    );

    let mut zero_arr = Array::zeros((0, records.ncols()));
    for n in 0..records.nrows() {
        zero_arr.push_row(records.row(n)).unwrap();
    }

    let data = Dataset::new(zero_arr, targets);

    Ok(data)
}

pub fn fit_linfa(frame: &DataFrame) {
    //should i put this into its own function?
    let nd_frame = frame.to_ndarray::<Float64Type>().unwrap();

    let data  = linfa_split(nd_frame).unwrap();

    let (train, test) = data.split_with_ratio(0.9);
    let model = LinearRegression::default().fit(&train).unwrap();

    let pred = model.predict(&test);
    let mse = pred.mean_squared_error(&test).unwrap();
    println!("{:?}", mse);

    if check_model_dir() {
        println!("\nSaving model");

        let reg_bytes = bincode::serialize(&model).expect("Issue serializing model");
        File::create("model/linfa_reg.model")
            .and_then(|mut f| f.write_all(&reg_bytes))
            .expect("Can not persist model");
    }
}

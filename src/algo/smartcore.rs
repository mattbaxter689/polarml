use polars::prelude::*;
use smartcore::metrics::mean_squared_error;
use smartcore::{
    linalg::basic::{
        arrays::{Array2, MutArray},
        matrix::DenseMatrix,
    },
    linear::linear_regression::LinearRegression,
    model_selection::train_test_split,
};

use std::fs::File;
use std::io::{Read, Write};

use crate::fileops::check_model_dir;

// Give shape andtype for data in frame

pub fn create_x_dense(x: &DataFrame) -> Result<DenseMatrix<f64>, PolarsError> {
    let nrows = x.height();
    let ncols = x.width();
    let x_array = x.to_ndarray::<Float64Type>().unwrap();

    let mut xmatrix: DenseMatrix<f64> = DenseMatrix::fill(nrows, ncols, 0.0);

    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in x_array.iter() {
        // Debug
        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();

        xmatrix.set((m_row, m_col), *val);
        // check what we have to update
        if m_col == ncols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    Ok(xmatrix)
}

//This function builds the regression model, so there shouldnt be a Need
//to return anything, I can just print out the accuracy and things
pub fn fit_smartcore(xmat: DenseMatrix<f64>, yvals: Vec<f64>) {
    
    let (x_train, x_test, y_train, y_test) = train_test_split(&xmat, &yvals, 0.2, true, Some(5));

    println!("Building the model");
    let model = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();

    let pred = model.predict(&x_test).unwrap();
    let mse = mean_squared_error(&y_test, &pred);
    println!("\n MSE: {:?}", mse);

    if check_model_dir() {
        println!("\nSaving model");

        let reg_bytes = bincode::serialize(&model).expect("Issue serializing model");
        File::create("model/lin_reg.model")
            .and_then(|mut f| f.write_all(&reg_bytes))
            .expect("Can not persist model");
    }
}

//Look into the model and its coefficients. Need to investigate if i can explore model more
pub fn investigate(path: String) {

    let lr_model: LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>> = {
        let mut buf: Vec<u8> = Vec::new();
        File::open(&path)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Can not load model");
        bincode::deserialize(&buf).expect("Can not deserialize the model")
    };

    println!("\nModel Parameters: ");
    println!("{:?}", lr_model.coefficients());
    println!("\n");
    println!("Model Intercept: ");
    println!("{:?}", lr_model.intercept());
}

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
use std::io::Write;

pub fn read_csv(path: &str) -> PolarsResult<DataFrame> {
    CsvReader::from_path(path)?.has_header(true).finish()
}

// Give shape andtype for data in frame
pub fn describe_df(df: &DataFrame) {
    println!("Dataframe shape:...");
    println!("{:?}", df.shape());

    println!("\n Dataframe schema:...");
    println!("{:?}", df.schema());

    println!("\n Dataframe:...");
    println!("{:?}", df);
}

// Get the feature and target variables separate
pub fn extract_feature_target(
    df: &DataFrame,
) -> (PolarsResult<DataFrame>, PolarsResult<DataFrame>) {
    let features = df.select(vec![
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "black",
        "lstat",
    ]);

    let target = df.select(["medv"]);

    return (features, target);
}

//TODO: I'm not too happy with this. Need to see if I can put this into array
// when reading into rust, instead of dataframe. Or make 2 different processes,
// one for description, and two for model building
pub fn create_x_mat(x: &DataFrame) -> Result<DenseMatrix<f64>, PolarsError> {
    let nrows = x.height();
    let ncols = x.width();
    let x_array = x.to_ndarray::<Float64Type>().unwrap();

    let mut xmatrix: DenseMatrix<f64> = DenseMatrix::fill(nrows, ncols, 0.0);
    // populate the matrix
    // initialize row and column counters
    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in x_array.iter() {
        // Debug
        //println!("{},{}", usize::try_from(row).unwrap(), usize::try_from(col).unwrap());
        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        // NB we are dereferencing the borrow with *val otherwise we would have a &val type, which is
        // not what set wants
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
pub fn build_regression(xmat: DenseMatrix<f64>, yvals: Vec<f64>) {
    //split dataf
    let (x_train, x_test, y_train, y_test) = train_test_split(&xmat, &yvals, 0.2, true, Some(5));

    println!("Building the model");
    //fit the model
    let model = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    println!("Model built");
    //alternatively
    // LienarRegression::fit(&x_train, &y_train,
    //    LinearRegressionParameters::default
    //    .with_solver(LinearRegressionSolverName::SVD)).unwrap();
    //gives more control over the type of solver you want to use (SVD preferred for this)
    //
    //Don't forget to also get the regression coefficients and possible plots of the model.
    // Could also look at comparing linfa with smartcore to see which is better.

    let pred = model.predict(&x_test).unwrap();
    let mse = mean_squared_error(&y_test, &pred);
    println!("MSE: {:?}", mse);

    println!("Saving model");
    //time to save the model, in case of future use
    //Could add a check to see if a model is already saved, and leave it be if is
    let reg_bytes = bincode::serialize(&model).expect("Issue serializing model");
    File::create("src/model/lin_reg.model")
        .and_then(|mut f| f.write_all(&reg_bytes))
        .expect("Can not persist model");
}

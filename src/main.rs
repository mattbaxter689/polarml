use clap::{Parser, Subcommand};
use polars::prelude::Float64Type;

mod algo;
mod fileops;

const CSV_FILE: &str = "data/housing.csv";

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Describe {
        #[arg(short, long, default_value = CSV_FILE)]
        path: String,
    },
    Smart {
        #[arg(short, long, default_value = CSV_FILE)]
        path: String,
    },
    Linfa {
        #[arg(short, long, default_value = CSV_FILE)]
        path: String,
    },
    // don't want default value here, since want user to specify model
    Info {
        #[arg(short, long)]
        path: String,
    },
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Some(Commands::Describe { path }) => {
            let df = fileops::read_csv(&path).unwrap();
            algo::schema::describe_df(&df);
        }
        Some(Commands::Smart { path }) => {
            let df = fileops::read_csv(&path).unwrap();
            let (x, y) = algo::schema::extract_feature_target(&df);
            let xs = x.as_ref().unwrap();
            let xdense = algo::smartcore::create_x_dense(&xs).unwrap();

            //set up y to be array
            let ydense = y.as_ref().unwrap().to_ndarray::<Float64Type>().unwrap();
            let mut target: Vec<f64> = Vec::new();
            for val in ydense.iter() {
                target.push(*val);
            }

            algo::smartcore::fit_smartcore(xdense, target);
        }
        Some(Commands::Linfa { path }) => {
            let df = fileops::read_csv(&path).unwrap();

            algo::linfa::fit_linfa(&df);
        }
        Some(Commands::Info { path }) => {
            algo::smartcore::investigate(path);
        }
        None => {
            println!("No subcommand used");
        }
    }
}

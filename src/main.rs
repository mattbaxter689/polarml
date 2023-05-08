use clap::Parser;
use polars::prelude::Float64Type;

mod algo;
mod fileops;

const CSV_FILE: &str = "src/data/housing.csv";

#[derive(Parser)]
#[clap(
    version = "1.0",
    author = "Matt Baxter",
    about = "A cli to help with clap and model building from command line"
)]

struct Cli {
    #[clap(subcommand)]
    command: Option<Commands>,
}

#[derive(Parser)]
enum Commands {
    Describe {
        #[clap(long, default_value = CSV_FILE)]
        path: String,
    },
    Fit {
        #[clap(long, default_value = CSV_FILE)]
        path: String,
    },
    Info {
        #[clap(long, default_value = "src/model/lin_reg.model")]
        path: String,
    },
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Some(Commands::Describe { path }) => {
            let df = fileops::read_csv(&path).unwrap();
            algo::operations::describe_df(&df);
        }
        Some(Commands::Fit { path }) => {
            let df = fileops::read_csv(&path).unwrap();
            let (x, y) = algo::operations::extract_feature_target(&df);
            let xs = x.unwrap();
            let xdense = algo::operations::create_x_dense(&xs).unwrap();

            //set up y to be array
            let ydense = y.unwrap().to_ndarray::<Float64Type>().unwrap();
            let mut target: Vec<f64> = Vec::new();
            for val in ydense.iter() {
                target.push(*val);
            }

            algo::operations::fit_smartcore(xdense, target);
        }
        Some(Commands::Info { path }) => {
            algo::operations::investigate(path);
        }
        None => {
            println!("No subcommand used");
        }
    }
}

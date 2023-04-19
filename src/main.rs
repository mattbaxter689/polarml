use clap::Parser;
use polars::prelude::Float64Type;

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
    Build {
        #[clap(long, default_value = CSV_FILE)]
        path: String,
    },
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Some(Commands::Describe { path }) => {
            let df = polarml::read_csv(&path).unwrap();
            polarml::describe_df(&df);
        }
        Some(Commands::Build { path }) => {
            let df = polarml::read_csv(&path).unwrap();
            let (x, y) = polarml::extract_feature_target(&df);
            let xs = x.unwrap();
            let xdense = polarml::create_x_mat(&xs).unwrap();

            //set up y to be array
            let ydense = y.unwrap().to_ndarray::<Float64Type>().unwrap();
            let mut target: Vec<f64> = Vec::new();
            for val in ydense.iter() {
                target.push(*val);
            }

            polarml::build_regression(xdense, target);
        }
        None => {
            println!("No subcommand used");
        }
    }
}

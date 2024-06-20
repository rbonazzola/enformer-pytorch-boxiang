def main(args):

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mlflow_experiment", "--expname", dest="mlflow_expname", type=str)       
    parser.add_argument("--mlflow_run_name", "--run_name", dest="mlflow_run_name", type=str)       
    parser.add_argument("--comments", type=str, help="Comments to be added to the MLflow run as a tag.")
    # parser.add_argument()

    args = parser.parse_args()

    main(args)
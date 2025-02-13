from src.setlogging import setup_logger
from src.apiconn import (OpenAiConnInfoCarrier, GeminiConnInfoCarrier, ApiConnInfoCarrier, LlmQuerier)

from dotenv import load_dotenv
import pandas as pd
import argparse
import logging
from ast import literal_eval

# Load environment variables
load_dotenv('llm_api_params.env')

logger, log_stream = setup_logger(logging.INFO)

def main(clsnms: list[str] = None):

    logger.info(f"Class names to process: {clsnms}")
    query = "What is U2's most popular song?"
    
    logger.info(f'Query to execute:{query}')
    if clsnms:
        ans_list = []
        logger.info("Processing class names.")
        for cls in clsnms:
            logger.info(f"Processing class name: {cls}")            
            lm_query_obj = LlmQuerier.get_lm_conn_obj(cls)
            ans = lm_query_obj.get_query_results(query)
            ans_list.append(ans)
    else:
        logger.error("No class names to process provided as input.")
        raise ValueError("No class names to process provided as input.")
    
    # Create a pandas dataframe to store the results
    df = pd.DataFrame({
        'Class Name': clsnms,
        'Answer': ans_list
    })
    logger.info(f"Results DataFrame:\n{df}")
    
#######################
# Invoke main.py
#######################
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query multiple LLMs.")
    parser.add_argument(
        '--class_names', 
        nargs='+',
        help="""List of class names to instantiate as string, e.g. "['class1','class2]" """
        )

    args = parser.parse_args()
    args.class_names = literal_eval(args.class_names[0])
    
    # Call main function
    main(args.class_names)
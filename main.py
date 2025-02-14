from src.setlogging import setup_logger
from src.apiconn import (OpenAiConnInfoCarrier, GeminiConnInfoCarrier, ApiConnInfoCarrier, LlmQuerier)

from dotenv import load_dotenv
import os
import argparse
import logging
from ast import literal_eval
import pandas as pd

# Load environment variables
load_dotenv('llm_api_params.env')

logger, log_stream = setup_logger(logging.INFO)

def main(clsnms: list[str] = None, query: str = None):

    logger.info(f"Class names to process: {clsnms}")
    
    if not clsnms:
        clsnms = ['OpenAiConnInfoCarrier', 'GeminiConnInfoCarrier']
    if not query:
        query = "What is U2's most popular song?"
    
    logger.info(f'Query to execute:{query}')
    if clsnms:
        logger.info("Processing class names.")
        
        ans_list = []
        for cls in clsnms:
            logger.info(f"Processing class name: {cls}")            
            lm_query_obj = LlmQuerier.get_lm_conn_obj(cls)
            # logger.info(f"Conn Info Params: {lm_query_obj.api_info_carrier.conn_params}")
            
            ans = lm_query_obj.get_query_results(query)
            ans_list.append(ans)
            logger.info(f'Language Model Ans: {ans}')
            
        # Create a pandas dataframe
        df = pd.DataFrame({
            'ClassName': clsnms,
            'Answer': ans_list
        })

        logger.info(f"Answers by Models:\n{df}")
    else:
        logger.error("No class names to process provided as input.")
        raise ValueError("No class names to process provided as input.")
    
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

    parser.add_argument(
        '--prompt',
        type=str,
        help="The prompt to query the LLMs with."
    )

    args = parser.parse_args()
    args.class_names = literal_eval(args.class_names[0])
    
    # Call main function
    main(args.class_names, args.prompt)
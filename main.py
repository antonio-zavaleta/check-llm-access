from src.setlogging import setup_logger

logger, log_stream = setup_logger()

def main():
    logger.info("Hello, World!")

if __name__ == "__main__":
    main()
import logging

def setup_logging(log_file='chatbot.log', log_level=logging.WARNING):
    try:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging setup completed successfully")
        return logger
    except Exception as e:
        print(f"Failed to configure logging: {str(e)}")
        logging.basicConfig(level=log_level)
        logger = logging.getLogger(__name__)
        return logger